import math
import random
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

# Configuration
SAFE_DISTANCE = 40      
KP_ROTATION = 2.0       
KP_FORWARD = 0.5        

class Pilot:
    def __init__(self, drone):

        self.drone = drone

    def is_blocked_by_drone(self, safety_dist=60.0, safety_angle=0.2):
        """Check if blocked by another drone (Used in Driver logic)."""
        semantic_data = self.drone.semantic_values()
        if not semantic_data: return False
        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if data.distance < safety_dist and abs(data.angle) < safety_angle:
                    return True
        return False
    
    def repulsive_force(self):

            repulsion_coeff = 20
            total_rad_repulsion = 0
            total_orthor_repulsion = 0

            lidar_data = self.drone.lidar_values()
            ray_angles = self.drone.lidar_rays_angles()

            for elt in range (180):

                if lidar_data[elt] < 220:

                    force = repulsion_coeff / lidar_data[elt] ** 2 
                    unit_vector_angle = ray_angles[elt] + math.pi

                    total_rad_repulsion += force * np.cos(unit_vector_angle)
                    total_orthor_repulsion += force *np.sin(unit_vector_angle)

            #total_orthor_repulsion = min(0.7, total_orthor_repulsion)
            #total_rad_repulsion = min(0.7, total_rad_repulsion)

            return (total_rad_repulsion, total_orthor_repulsion)
    
    def repulsive_force(self) -> tuple:
        
        repulsion_coeff = 45
        total_rad_repulsion = 0
        total_orthor_repulsion = 0
        nb_rays_wounded_pers = 90
        nb_rays_drone = nb_rays_wounded_pers // 6
        nb_rays_rescue_center = nb_rays_drone

        lidar_data = self.drone.lidar_values()
        ray_angles = self.drone.lidar_rays_angles()
        semantic_data = self.drone.semantic_values()
        semantic_data_bool = False

        priority:bool = self.drone.comms.get_priority()

        for elt in semantic_data: 

            if elt.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:

                if elt.distance == 0: force = nb_rays_wounded_pers * repulsion_coeff / 0.5 ** 2 

                else: force = nb_rays_wounded_pers * repulsion_coeff / elt.distance ** 2 

            elif DroneSemanticSensor.TypeEntity.RESCUE_CENTER and self.drone.state == 'RETURNING':

                if elt.distance == 0: force = nb_rays_rescue_center * repulsion_coeff / 0.5 ** 2 

                else: force = nb_rays_rescue_center * repulsion_coeff / elt.distance ** 2 

            else: force = 0

            if force != 0:

                angle_deg = round(np.rad2deg(normalize_angle(elt.angle, True)))
                index = angle_deg // 2

                for i in range(index - nb_rays_wounded_pers, index +nb_rays_wounded_pers  + 1):

                    lidar_data[i % 180] = 300

                unit_vector_angle = elt.angle

                total_rad_repulsion += force *np.cos(unit_vector_angle)

        total_rad_repulsion, total_orthor_repulsion = 0,0

        for elt in range (180):

            if lidar_data[elt] < 220 and lidar_data[elt] != 0:

                force = repulsion_coeff / lidar_data[elt] ** 2 
                unit_vector_angle = ray_angles[elt] + math.pi

                total_rad_repulsion += force * np.cos(unit_vector_angle)
                total_orthor_repulsion += force *np.sin(unit_vector_angle)

        #total_orthor_repulsion = min(0.7, total_orthor_repulsion)
        #total_rad_repulsion = min(0.7, total_rad_repulsion)

        #if priority: total_rad_repulsion,total_orthor_repulsion = 0,0

        #total_rad_repulsion = 0.7 * total_rad_repulsion

        #unstuck = self.unstuck()
        
        unstuck = None
#
        if unstuck is not None:
#
            unstuck_rad_force, unstuck_orthor_force = unstuck
#
            total_rad_repulsion += unstuck_rad_force
            total_orthor_repulsion += unstuck_orthor_force

        return (total_rad_repulsion, total_orthor_repulsion)


    def move_to_target_PID(self) -> CommandsDict:
        """
        Your ORIGINAL control function.
        """

        # If there is no target, stop completely
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Compute vector to target
        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # --------------------------------------------------
        # 1. Rotate toward the target (PID rotation control)
        # --------------------------------------------------
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.drone.estimated_angle)
        
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # --------------------------------------------------
        # 2. Forward movement (OLD LOGIC: slow & stable)
        #    Gradual braking when approaching target
        # --------------------------------------------------
        MAX_SPEED = 0.6
        MAX_SPEED_RESCUE = 1
        BRAKING_DIST = 150.0
        STOP_DIST = 15.0 

        if dist_to_target > BRAKING_DIST:
            forward_cmd = MAX_SPEED
        elif dist_to_target > STOP_DIST:
            forward_cmd = (dist_to_target / BRAKING_DIST) * MAX_SPEED
            forward_cmd = max(0.1, forward_cmd)
        else:
            forward_cmd = 0.05

        # --------------------------------------------------
        # 3. Rotation discipline
        #    Do NOT move forward if angle is too large
        #    (prevents sliding while turning)
        # --------------------------------------------------
        if abs(angle_error) > 0.2:
            forward_cmd = 0.0 

        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # --------------------------------------------------
        # Special logic when RETURNING (carrying a victim)
        # Faster movement to reduce rescue time
        # --------------------------------------------------
        if self.drone.grasped_wounded_persons():
            forward_cmd = MAX_SPEED_RESCUE
            if dist_to_target <= 60.0:
                forward_cmd = 0.45

        # Initialize default lateral command
        cmd_lateral = 0.0

        # --------------------------------------------------
        # 4. Drone collision avoidance (OLD deadlock resolution)
        # Hard safety layer to prevent drone-to-drone blocking
        # --------------------------------------------------
        #if forward_cmd > 0.05 and self.is_blocked_by_drone(safety_dist=60.0):
        #    forward_cmd = 0.0 
        #    cmd_lateral = -0.6 

        # --------------------------------------------------
        # 5. [NEW INTEGRATION] Add lateral repulsive force
        # Only affects sideways motion, not forward speed
        # --------------------------------------------------
        cmd_forward_repulsive, cmd_lateral = self.repulsive_force()
        
        if (cmd_forward_repulsive > 0 and forward_cmd > 0) or (cmd_forward_repulsive < 0 and forward_cmd < 0):

            if abs(cmd_forward_repulsive) > abs(forward_cmd):

                forward_cmd = cmd_forward_repulsive

        else:
            
            forward_cmd += cmd_forward_repulsive

        # Accumulate lateral forces
        #cmd_lateral += rep_lat
        
        # Clamp lateral command
        #cmd_lateral = max(-1.0, min(1.0, cmd_lateral))

        # --------------------------------------------------
        # Smart grasper logic
        # Automatically grab/release depending on state
        # --------------------------------------------------
        grasper_val = 0
        if self.drone.state in ["RETURNING", "DROPPING"]:
            grasper_val = 1
        elif self.drone.state == "RESCUING":
            if dist_to_target <= 50.0:
                grasper_val = 1
            else:
                grasper_val = 0
        elif self.drone.state == "END_GAME" and self.drone.grasped_wounded_persons():
             grasper_val = 1

        if self.drone.not_grapsed:
            grasper_val = 0

        # --------------------------------------------------
        # Anti-stuck mechanism (near Rescue Center walls)
        # Uses front LiDAR rays to detect close obstacles
        # If blocked, slide sideways to escape
        # --------------------------------------------------
        if self.drone.state in ["RETURNING", "END_GAME"] and dist_to_target < 100.0 and dist_to_target > 30.0:
            lidar_vals = self.drone.lidar_values()
            if lidar_vals is not None:
                front_rays = lidar_vals[85:95] 
                if len(front_rays) > 0:
                    min_front_dist = np.min(front_rays)
                    if min_front_dist < 15.0:
                        forward_cmd = 0.0
                        slide_force = 0.6 
                        
                        if abs(angle_error) < 0.1:
                            cmd_lateral = -slide_force 
                        elif angle_error > 0:
                            cmd_lateral = slide_force 
                        else:
                            cmd_lateral = -slide_force 

                        if forward_cmd > 0:

                            forward_cmd = min(MAX_SPEED, forward_cmd)

                        else:

                            forward_cmd = max(-MAX_SPEED, forward_cmd)

                        if cmd_lateral > 0:

                            cmd_lateral = min(MAX_SPEED, cmd_lateral)

                        else:

                            cmd_lateral = max(-MAX_SPEED, cmd_lateral)

                        return {
                            "forward": forward_cmd,
                            "lateral": cmd_lateral,
                            "rotation": rotation_cmd, 
                            "grasper": grasper_val
                        }

        if forward_cmd > 0:

            forward_cmd = min(MAX_SPEED, forward_cmd)

        else:

            forward_cmd = max(-MAX_SPEED, forward_cmd)

        if cmd_lateral > 0:

            cmd_lateral = min(MAX_SPEED, cmd_lateral)

        else:

            cmd_lateral = max(-MAX_SPEED, cmd_lateral)


        # Final command output
        return {
            "forward": forward_cmd, 
            "lateral": cmd_lateral, 
            "rotation": rotation_cmd, 
            "grasper": grasper_val
        }


    def unstuck(self) -> None:

        if self.drone.same_position_timestep < 10: return (0,0)

        lidar_data = self.drone.lidar_values()
        ray_angles = self.drone.lidar_rays_angles()

        unstuck_velocity = 0.5
        unstuck_angle = np.deg2rad(15) # angle between wall and direction the drone is going in to unstuck
        nb_rays_around = 5

        current_pos = self.drone.estimated_pos
        current_target = self.drone.current_target  

        if self.drone.current_target is None: return (0,0)

        relative_angle = np.atan2(current_target[1], current_target[0])
        relative_dist = np.linalg.norm((current_pos[0] - current_target[0], current_pos[1] - current_target[1]))

        #print(relative_angle, relative_dist)

        corresponding_index = round(np.rad2deg(relative_angle + math.pi)) // 2
        stuck = False
        l_stuck = []

        for index in range(corresponding_index - nb_rays_around ,corresponding_index + nb_rays_around + 1):

            if lidar_data[index % 180] < relative_dist + SAFE_DISTANCE:

                stuck = True
                l_stuck.append(index)

        if stuck: 
            list_possible_paths = self.drone.nav.lidar_possible_paths()
            if list_possible_paths:
                self.drone.current_target = list(list_possible_paths[-1][0])
            return (0.8,0.8)

            #good_left = True
            #for index in range(l_stuck[0] - 2 * nb_rays_around - 1, l_stuck[0]):
#
            #    if lidar_data[index % 180] < relative_dist + SAFE_DISTANCE:
#
            #        good_left = False
#
            #if good_left: 
#
            #    unstuck_rad_force = - unstuck_velocity * np.sin(unstuck_angle)
            #    unstuck_orthor_force = unstuck_velocity * np.cos(unstuck_angle)
            #    
            #    return unstuck_rad_force, unstuck_orthor_force                
#
            #good_right = True
            #for index in range(l_stuck[-1] + 1, l_stuck[-1] + 2 * nb_rays_around + 2):
#
            #    if lidar_data[index % 180] < relative_dist + SAFE_DISTANCE:
#
            #        good_right = False
#
            #if good_right: 
            #    
            #    unstuck_rad_force = - unstuck_velocity * np.sin(unstuck_angle)
            #    unstuck_orthor_force = - unstuck_velocity * np.cos(unstuck_angle)
            #    
            #    return unstuck_rad_force, unstuck_orthor_force
#
            ## we didn't manage to unstuck by moving left or right
            #list_possible_paths = self.drone.nav.lidar_possible_paths()
            #if list_possible_paths:
            #    self.drone.current_target = list(list_possible_paths[-1][0])
            #    print(self.drone.current_target)
            
        return (0,0)
    
