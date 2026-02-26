import math
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from random import random

class Pilot:
    """
    Low-level Flight Controller.
    
    Strategy Update: "Soft Approach"
    - Variable Rotation Gain (KP_ROT): High when far, Low when near (damped).
    - Continuous Motion: Never stop completely to align; maintain momentum.
    - Relaxed Grasping: Allow grasping even with slight misalignment.
    """
    def __init__(self, drone):
        """Store the drone reference and initialize controller memory."""
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
        side_sign = 1.0 if (self.drone.identifier is None or self.drone.identifier % 2 == 0) else -1.0

        align_threshold = math.radians(12.0)
        grasp_distance = 34.0

        # Overtake mode: if we are almost perfectly behind a moving victim, push harder
        # and add a tiny lateral bias so we do not stay in the exact same lane forever.
        if abs_error < math.radians(11.0) and wounded.distance > 24.0:
            overtake_speed = float(np.clip(0.74 + 0.005 * wounded.distance, 0.74, 1.0))
            return self.move_function(forward = overtake_speed, lateral = float(0.24 * side_sign), rotation = float(np.clip(2.8 * angle_error, -0.9, 0.9)), grasper = 0, repulsive_force_bool = True)

        # Keep moving while aligning so we do not trail behind moving victims.
        if abs_error > align_threshold:
            moving_align_speed = float(np.clip(0.44 + 0.006 * wounded.distance, 0.44, 0.92))
            return self.move_function(forward = moving_align_speed, lateral = 0, rotation = float(np.clip(3.4 * angle_error, -1.0, 1.0)), grasper = 0, repulsive_force_bool = True)

        if wounded.distance > grasp_distance:
            approach_speed = float(np.clip(0.52 + 0.009 * (wounded.distance - grasp_distance), 0.52, 1.0))
            return self.move_function(forward = approach_speed, lateral = 0, rotation = float(np.clip(2.2 * angle_error, -0.75, 0.75)), grasper = 0, repulsive_force_bool = True)

        return self.move_function(forward = 0.14, lateral = 0, rotation = float(np.clip(2.0 * angle_error, -0.7, 0.7)), grasper = 1, repulsive_force_bool = True)

    def low_battery(self) -> None:
        '''
        Takes care of the returning when little timesteps are left. Mainly changes drone states
        
        :param self: self
        :return: None
        '''

        if self.drone.steps_remaining <= self.drone.RETURN_TRIGGER_STEPS:
            if self.drone.state not in ["RETURNING", "DROPPING", "END_GAME"]:
                print(f"[{self.drone.identifier}] 🔋 LOW BATTERY. Returning.")
                self.drone.state = "RETURNING"
                self.drone.current_target = None
            if self.drone.is_inside_return_area and not self.drone.grasped_wounded_persons(): self.drone.state = "END_GAME"

    def is_in_tight_area(self,SMALL_VALUE:int = 50,min_nb_rays:int = 30):
        '''
        returns whether or not the drone is in a tight area (and should get a weaker repulsive force)

        :param self: self
        :param SMALL_VALUE: The threshold below which a ray contributes to the drone being considered in a tight area
        :type SMALL_VALUE: int
        :param min_nb_rays: Minimal number of rays for us to consider the drone in a tight area
        :type min_nb_rays: int
        :return: Do more than min_nb_rays hit an obstacle closer than SMALL_VALUE?
        :rtype: bool
        '''

        lidar_data = self.drone.lidar_values()
        nb = 0
        nb_rays_practical = min_nb_rays // 5
        for ray in range(0,nb_rays_practical,5):

            if lidar_data[ray] < SMALL_VALUE: nb += 1
            if nb > min_nb_rays: return True

        return False

    def repulsive_force(self, total_correction_norm:float = 0.8) -> tuple:

        '''
        This function handles collision evading of drones, be it with walls or with other drones. It also takes into account rescuing and returning states so as to make repulsion force weaker
                
        :param self: self
        :param total_correction_norm: a coefficient by which we multiply the force at the end to scale it down. given value is determined empirically
        :type total_correction_norm: float
        :return: (total_rad_correction, total_ortho_correction) that we add to the forward and lateral movement to get the right components
        :rtype: tuple
        '''

        total_rad_repulsion = 0
        total_orthor_repulsion = 0

        if self.is_in_tight_area():
            WALL_CONSTANT = 0.02
        else:
            WALL_CONSTANT = 0.007
        DRONE_CONSTANT = .1

        quotient_rad_repulsion = 4
        quotient_ortho_repulsion = 2
        min_angle_difference = 0.35 # (radian) the angle difference below which we assume that two semantic sensor rays hitting drones are actually hitting the same drone
        exponent_wall = 2.3
        exponent_drone = 2.3
        epsilon_range = 0.2 # when evading a drone, we add a random epsilon angle to our deviation, it is chosen in (-epsilon_range, epsilon_range)
        rays_discard_lidar_drone = 2
        end_game_coefficient = 0.1

        lidar_data = self.drone.lidar_values()
        semantic_data = self.drone.semantic_values()
        ray_angles = self.drone.lidar_rays_angles()
        last_drone_angle = None # prevents us from computing repulsive force for a drone multiple times
        list_rays_rescue = []

        for elt in range(len(semantic_data)):

            if semantic_data[elt].entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not semantic_data[elt].grasped and self.drone.state =='RESCUING':

                corresponding_index = round(np.rad2deg(semantic_data[elt].angle)) // 2 + 90
                for ray in range(corresponding_index - round(2 * (100 / semantic_data[elt].distance)), corresponding_index + round(2 * (100 / semantic_data[elt].distance)) + 1):

                    lidar_data[ray % 180] = 400
                

            if semantic_data[elt].entity_type == DroneSemanticSensor.TypeEntity.DRONE:

                if last_drone_angle is not None and abs(last_drone_angle - semantic_data[elt].angle) < min_angle_difference: continue

                last_drone_angle = semantic_data[elt].angle # prevents us from computing force multiple times if multiple rays hit the same drone




                # Need to come back to this (MARC)




                if not self.drone.grasped_wounded_persons(): # drones who have priority don't deviate from their trajectories because of other drones




                # Need to come back to this (MARC)




                    force = DRONE_CONSTANT / (semantic_data[elt].distance / 100) ** exponent_drone
                    unit_vector_angle = semantic_data[elt].angle + np.pi + epsilon_range * (2 * random() - 1) # unstuck the drones by adding a small random deviation angle
                    
                    total_rad_repulsion += force * np.cos(unit_vector_angle)
                    total_orthor_repulsion += force *np.sin(unit_vector_angle)
                index = round(np.deg2rad(semantic_data[elt].angle)) // 2 + 90
                for i in range(index - rays_discard_lidar_drone, index + rays_discard_lidar_drone + 1):

                    lidar_data[i % 180] = 400

        # Rescue center should not exert a force on a drone that is rescuing
            if semantic_data[elt].entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and self.drone.grasped_wounded_persons(): 

                if list_rays_rescue == []: list_rays_rescue = [semantic_data[elt].angle]
                elif len(list_rays_rescue) == 1: list_rays_rescue.append(semantic_data[elt].angle)
                else:
                    list_rays_rescue[-1] = semantic_data[elt].angle

        if len(list_rays_rescue) == 2:

            corresponding_first_index = round(np.rad2deg(list_rays_rescue[0])) // 2 + 90
            corresponding_last_index = round(np.rad2deg(list_rays_rescue[1])) // 2 + 90
            if corresponding_last_index < corresponding_first_index: corresponding_last_index += 180

            for ray in range(corresponding_first_index, corresponding_last_index + 1): 

                lidar_data[ray % 180] = 400

        # repulsive force from walls
        for elt in range (len(lidar_data)):

            if 0 < lidar_data[elt] < 200:

                force = WALL_CONSTANT / (lidar_data[elt] / 100) ** exponent_wall
                unit_vector_angle = ray_angles[elt] + math.pi

                total_rad_repulsion += (force / quotient_rad_repulsion) * np.cos(unit_vector_angle)
                total_orthor_repulsion += (force / quotient_ortho_repulsion) * np.sin(unit_vector_angle)

        # scales down the repulsive force
        total_rad_repulsion *= total_correction_norm
        total_orthor_repulsion *= total_correction_norm

        if self.drone.state == 'END_GAME' and self.drone.is_inside_return_area:

            total_rad_repulsion *= end_game_coefficient
            total_orthor_repulsion *= end_game_coefficient 
        
        return total_rad_repulsion, total_orthor_repulsion


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
            corr1, corr2 = self.repulsive_force(total_correction_norm = total_correction_norm)
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
            return self.drone.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)
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

        # 5. Active Braking & Approach
        BRAKE_DIST = 120.0 
        if dist_to_target < BRAKE_DIST:
            # Keep enough minimum thrust close to target to avoid stall near drop zone.
            base_forward = max(0.4, dist_to_target * 0.03) 
            base_forward *= alignment_factor
            if self.current_speed > 4.0: base_forward = -0.4 
        else:
            base_forward = self.drone.MAX_SPEED * alignment_factor

        # Catch-up boost while rescuing moving targets.
        if self.drone.state == "RESCUING" and not self.drone.grasped_wounded_persons():
            wounded = self._nearest_visible_wounded()
            if wounded is not None and 18.0 < wounded.distance < 170.0:
                # Stronger boost at medium range, lighter near latch distance.
                if wounded.distance > 70.0:
                    base_forward = min(1.0, base_forward + 0.16)
                else:
                    base_forward = min(1.0, base_forward + 0.10)

        # Dampen steering oscillations in tight corridors without reducing thrust.
        lidar_values = self.drone.lidar_values()
        if lidar_values is not None and min(lidar_values) < 60.0:
            if abs(angle_error) < 0.5:
                rotation_cmd *= 0.5

        # Keep wall-shield behavior consistent when reversing.
        if not is_reversing:
            # Forward motion: keep thrust and repulsion signs unchanged.
            forward_cmd = base_forward + repulsion_rad
        else:
            # Reverse motion: flip thrust only, keep wall-repulsion direction.
            forward_cmd = -base_forward + repulsion_rad
            
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        # 7. Front-approach grasp logic during rescue
        front_grasp_cmd = self.front_grasp_alignment_command()
        if front_grasp_cmd is not None:
            #print(f'{self.drone.identifier} {self.drone.state}front_grasp_alignment_command_pilot')
            return front_grasp_cmd

        grasper_val = 1 if self.drone.grasped_wounded_persons() else 0
        #print(forward_cmd, np.clip(cmd_lateral, -1.0, 1.0), rotation_cmd, grasper_val)
        #print(f'{self.drone.identifier} {self.drone.state} move_to_target_carrot_last_pilot')
        return self.move_function(forward = forward_cmd, lateral = 0, rotation = rotation_cmd, grasper = grasper_val, repulsive_force_bool = True)
    

    def dispersion(self):

        '''Handles the event where the drone's state is "DISPERSING": evades other drones and eventually switches to "EXPLORING"
        
        :return: Calls move function or changes state
        :rtype: CommandsDict
        '''
        # Initialize a memory variable to track when safe distance is reached
            
        dx = self.drone.estimated_pos[0] - self.drone.initial_position[0]
        dy = self.drone.estimated_pos[1] - self.drone.initial_position[1]
        dist_moved = math.hypot(dx, dy)
        
        # Check if we have fulfilled the escape conditions
        if self.drone.safe_dispersion_reached_tick is None:
            # Condition: Run for at least 50 ticks AND exceed 150px safe distance.
            # Safety Timeout: Force escape at 200 ticks to prevent infinite deadlock.
            if (self.drone.cnt_timestep >= 50 and dist_moved >= 150.0) or self.drone.cnt_timestep >= 1:
                self.drone.safe_dispersion_reached_tick = self.drone.cnt_timestep
        
        # 1. Active Repulsion: Push away from peers, walls, and Rescue Center
        if self.drone.safe_dispersion_reached_tick is None:
            if self.drone.print_move_functions: print(f'{self.drone.identifier} {self.drone.state} dispersion move_function 1 pilot')
            return self.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)

        
        # 2. Scanning Mode: Spin to map surroundings (Rescue Center) with Lidar
        elif self.drone.cnt_timestep < self.drone.safe_dispersion_reached_tick + 50: 
            if self.drone.print_move_functions: print(f'{self.drone.identifier} {self.drone.state} dispersion move_function 21 pilot')
            return self.move_function(forward=0, lateral=0, rotation=1, grasper=0, repulsive_force_bool=True)
        
        # 3. Lock Natural Escape Angle & Transition to Exploring
        else:
            if dist_moved > 5.0:
                self.drone.preferred_angle = math.atan2(dy, dx)
            else: 
                self.drone.preferred_angle = self.drone.estimated_angle
                
            deg_angle = math.degrees(self.drone.preferred_angle)
            print(f"[{self.drone.identifier}] 🚀 WARMUP DONE. Dist: {dist_moved:.1f}px. Angle: {deg_angle:.1f}°. EXPLORING!")
            self.drone.state = "EXPLORING"