import math
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class Pilot:
    """
    Low-level Flight Controller.
    
    Strategy Update: "Soft Approach"
    - Variable Rotation Gain (KP_ROT): High when far, Low when near (damped).
    - Continuous Motion: Never stop completely to align; maintain momentum.
    - Relaxed Grasping: Allow grasping even with slight misalignment.
    """
    def __init__(self, drone):
        self.drone = drone
        self.last_pos = None
        self.current_speed = 0.0

    def _nearest_visible_wounded(self):
        """Returns nearest currently visible non-grasped wounded person."""
        semantic_data = self.drone.semantic_values()
        if not semantic_data:
            return None

        nearest = None
        best_dist = float("inf")
        for data in semantic_data:
            if data.entity_type != DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                continue
            if data.grasped:
                continue
            if data.distance < best_dist:
                best_dist = data.distance
                nearest = data
        return nearest

    def front_grasp_alignment_command(self):
        """
        During RESCUING, rotate to face the wounded person first, then grasp.
        """
        if self.drone.grasped_wounded_persons():
            return None
        if self.drone.state != "RESCUING":
            return None

        wounded = self._nearest_visible_wounded()
        if wounded is None:
            return None

        angle_error = normalize_angle(wounded.angle)
        abs_error = abs(angle_error)

        align_threshold = math.radians(10.0)
        grasp_distance = 32.0

        if abs_error > align_threshold:
            return self.move_function(forward = 0, lateral = 0, rotation = float(np.clip(3.0 * angle_error, -1.0, 1.0)), grasper = 0, repulsive_force_bool = True)

        if wounded.distance > grasp_distance:
            approach_speed = float(np.clip(0.15 + 0.006 * (wounded.distance - grasp_distance), 0.15, 0.45))
            return self.move_function(forward = approach_speed, lateral = 0, rotation = float(np.clip(2.0 * angle_error, -0.6, 0.6)), grasper = 1, repulsive_force_bool = True)

        return self.move_function(forward = 0.6, lateral = 0, rotation = float(np.clip(2.0 * angle_error, -0.6, 0.6)), grasper = 1, repulsive_force_bool = True)
    
    def calculate_repulsive_force(self):
        """Calculates repulsive force to avoid colliding with other drones."""
        total_lat = 0.0
        semantic_data = self.drone.semantic_values()
        if not semantic_data: return 0.0, 0.0

        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                dist = data.distance
                if 0.1 < dist < 150.0:
                    K = 500.0 
                    force_magnitude = K / (dist ** 2)
                    force_magnitude = min(1.5, force_magnitude)
                    push_angle = data.angle + math.pi - 0.5 
                    total_lat += force_magnitude * math.sin(push_angle)
        return 0.0, total_lat

    def calculate_wall_repulsion(self, aggressive: bool = False, angle_error: float = 0.0):
        """Calculates repulsive forces from walls using Lidar."""
        lidar = self.drone.lidar_values()
        angles = self.drone.lidar_rays_angles()
        if lidar is None or angles is None: return 0.0, 1.0 

        total_lat = 0.0
        min_dist_detected = 300.0

        if aggressive:
            K_wall = 120.0; ignore_dist = 40.0; critical_dist = 10.0; slow_down_threshold = 40.0
        else:
            K_wall = 350.0; ignore_dist = 80.0; critical_dist = 20.0; slow_down_threshold = 60.0

        # Smart Repulsion: Reduce push if aiming at a gap
        if abs(angle_error) < 0.2:
            K_wall *= 0.4

        step = 5 
        for i in range(0, len(lidar), step):
            dist = lidar[i]
            if 10.0 < dist < ignore_dist:
                if dist < min_dist_detected: min_dist_detected = dist

                force = K_wall / (dist ** 1.8)
                force = min(1.0, force) 

                angle_obs = angles[i]
                push_angle = angle_obs + math.pi 
                total_lat += force * math.sin(push_angle)

        speed_factor = np.clip((min_dist_detected - critical_dist) / slow_down_threshold, 0.3, 1.0)
        if aggressive: speed_factor = max(0.5, speed_factor)

        return total_lat, speed_factor

    def low_battery(self) -> None:
        '''
        Takes care of the returning when little timesteps are left. Mainly changes drone states
        
        :param self: self
        :return: None
        '''

        if self.drone.steps_remaining <= self.drone.RETURN_TRIGGER_STEPS:
            if not self.drone.grasped_wounded_persons():
                if self.drone.is_inside_return_area: self.drone.state = "END_GAME"
            else:
                if self.drone.state not in ["RETURNING", "DROPPING", "END_GAME"]:
                    print(f"[{self.drone.identifier}] 🔋 LOW BATTERY. Returning.")
                    self.drone.state = "RETURNING"
                    self.drone.current_target = None 


    def repulsive_force(self, total_correction_norm:float = 0.7) -> tuple:
        '''
        returns a radial and an orthoradial component of a repulsive force that helps with preventing collisions with surroundings
        
        :param self: self
        :param total_correction_norm: An (optional) float describing the force exerted on the drone
        :type force: int
        :return: (radial, orthoradial)
        :rtype: tuple
        '''
        total_rad_repulsion = 0
        total_orthor_repulsion = 0

        lidar_data = self.drone.lidar_values()
        semantic_data = self.drone.semantic_values()
        ray_angles = self.drone.lidar_rays_angles()

        for elt in range (180):

            if lidar_data[elt] < 220:
                if lidar_data[elt] == 0: lidar_data[elt] = 0.1

                force = 1 / lidar_data[elt] ** 2 
                unit_vector_angle = ray_angles[elt] + math.pi

                total_rad_repulsion += force * np.cos(unit_vector_angle)
                total_orthor_repulsion += force *np.sin(unit_vector_angle)

        for elt in semantic_data:

            if (elt.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and self.drone.state == 'RESCUING') or (self.drone.state == 'RETURNING' and elt.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):

                force = 1/10

                total_rad_repulsion += force * np.cos(elt.angle)
                total_orthor_repulsion += force *np.sin(elt.angle)

            if elt.entity_type == DroneSemanticSensor.TypeEntity.DRONE:

                if elt.distance == 0: force = 100
                else: force = 4/elt.distance ** 2
     
                unit_vector_angle = elt.angle + math.pi

                total_rad_repulsion += force * np.cos(unit_vector_angle)
                total_orthor_repulsion += force *np.sin(unit_vector_angle)

        #total_orthor_repulsion = min(0.7, total_orthor_repulsion)
        #total_rad_repulsion = min(0.7, total_rad_repulsion)

        actual_norm_correction = math.hypot(total_rad_repulsion,total_orthor_repulsion)
        if actual_norm_correction < 0.001: return 0,0

        total_rad_repulsion *= total_correction_norm / actual_norm_correction
        total_orthor_repulsion *= total_correction_norm / actual_norm_correction

        print(total_rad_repulsion, total_orthor_repulsion)

        return (total_rad_repulsion, total_orthor_repulsion)

    def move_function(self,forward: float, lateral: float, rotation: float, grasper: int, repulsive_force_bool:bool,total_correction_norm:float = 0.8) -> CommandsDict:
        
        '''
        returns a CommandsDict for a drone standing still, will still move (slowly) a bit just to avoid obstacles
        
        :param self: self
        :param forward: forward velocity in (-1,1). Negative values indicate backwards movement
        :type forward: float
        :param lateral: lateral velocity in (-1,1)
        :type lateral: float
        :param rotation: rotational velocity in (-1,1)
        :type rotation: float
        :param grasper: whether or not we are currently grasping (0 or 1)
        :type grasper: int
        :param repulsive_force_bool: whether or not to generate evasive repulsive force
        :type repulsive_force_bool: bool
        :param total_correction_norm: norm of the repulsive (correction) force
        :type total_correction_norm: float
        :return: {"forward", "lateral", "rotation", "grasper"}
        :rtype: CommandsDict
        '''
        
        if repulsive_force_bool:
            corr1, corr2 = self.repulsive_force(total_correction_norm = total_correction_norm) # soft movement

            forward += corr1
            lateral += corr2

        if forward > 1: forward = 1
        elif forward < -1: forward = -1

        if lateral > 1: lateral = 1
        elif lateral < -1: lateral = -1


        return {"forward": forward, "lateral": lateral, "rotation": rotation, "grasper":grasper}

    def move_to_target_carrot(self) -> CommandsDict:
        '''
        Main control loop.
        :param self: self
        :return: Description
        :rtype: CommandsDict
        '''
        if self.drone.current_target is None:
            return self.drone.pilot.stand_still(grasper = False)

        # 1. Estimate Speed
        if self.last_pos is not None:
            move_dist = np.linalg.norm(self.drone.estimated_pos - self.last_pos)
            self.current_speed = move_dist 
        self.last_pos = self.drone.estimated_pos.copy()

        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)
        target_angle = math.atan2(delta_y, delta_x)

        is_reversing = (self.drone.grasped_wounded_persons() and self.drone.state == 'RETURNING')

        # State Checks
        is_final_approach = (self.drone.state == 'RESCUING' and dist_to_target < 55.0)
        is_aggressive = (self.drone.state == 'RESCUING') or (dist_to_target < 80.0)

        if is_reversing:
            desired_angle = normalize_angle(target_angle + math.pi)
        else:
            desired_angle = target_angle

        angle_error = normalize_angle(desired_angle - self.drone.estimated_angle)

        # =========================================================
        # 2. ROTATION CONTROL (VARIABLE KP STRATEGY)
        # =========================================================

        # Default: Fast response for navigation
        KP_ROT = 4.0 

        # [USER STRATEGY]: If close to target (< 70cm), reduce KP
        # This makes the drone turn softer/slower, avoiding oscillations/jitter close to the victim.
        if dist_to_target < 70.0:
            KP_ROT = 1.5 

        rotation_cmd = KP_ROT * angle_error
        rotation_cmd = np.clip(rotation_cmd, -1.0, 1.0)

        # 3. Wall Avoidance
        repulsion_rad, repulsion_orthor = self.repulsive_force()

        if is_final_approach:
            repulsion_orthor = 0.0          
            repulsion_rad = 0.5

        # Reduce speed if turning, but keep it smoother (cos^2 instead of cos^5)
        alignment_factor = max(0.2, math.cos(angle_error) ** 2)

        forward_cmd = self.drone.MAX_SPEED * alignment_factor + repulsion_rad

        if forward_cmd > 1: forward_cmd = 1
        elif forward_cmd < -1: forward_cmd = -1

        # 5. Active Braking & Approach
        BRAKE_DIST = 120.0 
        if dist_to_target < BRAKE_DIST:
            # [CRITICAL]: Never allow speed to drop to 0.0 unless ON the target
            # Keep at least 0.15 to "nudge" closer or push through soft collisions.
            forward_cmd = max(0.15, dist_to_target * 0.03) 

            # Apply alignment factor to slow down further if turning sharply
            forward_cmd *= alignment_factor

            # Reverse braking only if too fast
            if self.current_speed > 4.0: forward_cmd = -0.4 

        if is_reversing: forward_cmd = -forward_cmd
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        if abs(angle_error) > 0.5 and not is_reversing:
            repulsion_orthor += -0.5 * np.sign(angle_error)

        # 7. Front-approach grasp logic during rescue
        front_grasp_cmd = self.front_grasp_alignment_command()
        if front_grasp_cmd is not None:
            return front_grasp_cmd

        grasper_val = 1 if self.drone.grasped_wounded_persons() else 0
        #print(forward_cmd, np.clip(cmd_lateral, -1.0, 1.0), rotation_cmd, grasper_val)

        return self.move_function(forward = forward_cmd, lateral = np.clip(repulsion_orthor, -1.0, 1.0), rotation = rotation_cmd, grasper = grasper_val, repulsive_force_bool = True)