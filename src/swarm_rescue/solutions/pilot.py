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
        grasp_distance = 20

        if abs_error > align_threshold:
            return self.move_function(forward = 0, lateral = 0, rotation = float(np.clip(3.0 * angle_error, -1.0, 1.0)), grasper = 0, repulsive_force_bool = True)

        if wounded.distance > grasp_distance:
            approach_speed = 0.5    
            return self.move_function(forward = approach_speed, lateral = 0, rotation = float(np.clip(2.0 * angle_error, -0.6, 0.6)), grasper = 0, repulsive_force_bool = True, total_correction_norm = 0.2)

        return self.move_function(forward = 0.06, lateral = 0, rotation = float(np.clip(2.0 * angle_error, -0.6, 0.6)), grasper = 1, repulsive_force_bool = True)

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

    def repulsive_force(self, total_correction_norm: float) -> tuple:
        """
        Calculates the evasive repulsive forces (radial and orthoradial) to avoid collisions 
        with walls and other drones.
        
        Features:
        - VIP Hierarchical Yielding: Normal drones yield heavily to VIP drones carrying victims.
        - Anti-Crush Wall Shield: Exponentially spikes wall repulsion if pushed too close to a wall.
        
        Args:
            total_correction_norm (float): A scaling coefficient applied at the end.
            
        Returns:
            tuple: (total_rad_correction, total_ortho_correction)
        """
        total_rad_repulsion = 0
        total_orthor_repulsion = 0

        # Base wall constants (kept small to allow smooth navigation in open/tight spaces)
        if self.is_in_tight_area():
            BASE_WALL_CONSTANT = 0.02
        else:
            BASE_WALL_CONSTANT = 0.007
            
        DRONE_CONSTANT = .1
        quotient_rad_repulsion = 4
        quotient_ortho_repulsion = 2
        min_angle_difference = 0.35 
        exponent_wall = 2.3
        exponent_drone = 2.3
        epsilon_range = 0.2 
        rays_discard_lidar_drone = 2
        end_game_coefficient = 0.1

        lidar_data = self.drone.lidar_values()
        semantic_data = self.drone.semantic_values()
        ray_angles = self.drone.lidar_rays_angles()
        last_drone_angle = None 
        list_rays_rescue = []

        # Determine if THIS drone is a VIP (currently rescuing/carrying someone)
        am_i_vip = self.drone.grasped_wounded_persons()

        for elt in range(len(semantic_data)):

            # 1. MASK WOUNDED PERSONS
            if semantic_data[elt].entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not semantic_data[elt].grasped and self.drone.state =='RESCUING':
                corresponding_index = round(np.rad2deg(semantic_data[elt].angle)) // 2 + 90
                for ray in range(corresponding_index - round(2 * (100 / semantic_data[elt].distance)), corresponding_index + round(2 * (100 / semantic_data[elt].distance)) + 1):
                    lidar_data[ray % 180] = 400
                
            # 2. DRONE EVASION & VIP LOGIC
            if semantic_data[elt].entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if last_drone_angle is not None and abs(last_drone_angle - semantic_data[elt].angle) < min_angle_difference: 
                    continue
                last_drone_angle = semantic_data[elt].angle 

                is_other_vip = False
                
                # Check Comms to see if the observed drone is in the VIP list
                if hasattr(self.drone, 'comms') and hasattr(self.drone.comms, 'list_vip_drones'):
                    angle_global = self.drone.estimated_angle + semantic_data[elt].angle
                    obs_x = self.drone.estimated_pos[0] + semantic_data[elt].distance * math.cos(angle_global)
                    obs_y = self.drone.estimated_pos[1] + semantic_data[elt].distance * math.sin(angle_global)
                    
                    for vip_pos in self.drone.comms.list_vip_drones:
                        if math.hypot(obs_x - vip_pos[0], obs_y - vip_pos[1]) < 40.0:
                            is_other_vip = True
                            break

                compute_force = False
                force_multiplier = 1.0

                # Priority Rules:
                if not am_i_vip:
                    compute_force = True
                    if is_other_vip:
                        force_multiplier = 2.0  # Normal yields heavily to VIP
                else:
                    if is_other_vip:
                        compute_force = True    # VIP vs VIP symmetric evasion
                        force_multiplier = 1.0
                    else:
                        compute_force = False   # VIP ignores Normal drones

                if compute_force:
                    force = (DRONE_CONSTANT * force_multiplier) / (semantic_data[elt].distance / 100) ** exponent_drone
                    unit_vector_angle = semantic_data[elt].angle + math.pi + epsilon_range * (2 * random() - 1) 
                    
                    total_rad_repulsion += force * np.cos(unit_vector_angle)
                    total_orthor_repulsion += force * np.sin(unit_vector_angle)

                # Mask the drone from Lidar
                index = round(np.deg2rad(semantic_data[elt].angle)) // 2 + 90
                for i in range(index - rays_discard_lidar_drone, index + rays_discard_lidar_drone + 1):
                    lidar_data[i % 180] = 400

            # 3. MASK RESCUE CENTER (If rescuing)
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

        # 4. WALL REPULSION WITH ANTI-CRUSH INSTINCT
        for elt in range(len(lidar_data)):
            dist = lidar_data[elt]
            if 0 < dist < 200:
                
                # --- [NEW] ANTI-CRUSH SURVIVAL INSTINCT ---
                # Dynamically spike the wall force if pushed dangerously close to the wall.
                if dist < 25.0:
                    effective_wall_constant = BASE_WALL_CONSTANT * 10.0 # Emergency bounce!
                elif dist < 50.0:
                    effective_wall_constant = BASE_WALL_CONSTANT * 3.0  # Strong warning
                else:
                    effective_wall_constant = BASE_WALL_CONSTANT        # Normal smooth flight
                
                check_victim_around = False
                for data in self.drone.semantic_values():
                    if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON: check_victim_around = True
                if self.drone.state == "RESCUING" and check_victim_around: effective_wall_constant = 0
                force = effective_wall_constant / (dist / 100) ** exponent_wall
                
                angle_lidar = normalize_angle(ray_angles[elt])
                unit_vector_angle = angle_lidar + math.pi

                rad_push = (force / quotient_rad_repulsion) * np.cos(unit_vector_angle)
                orthor_push = (force / quotient_ortho_repulsion) * np.sin(unit_vector_angle)

                # --- [NEW] FORWARD MOMENTUM PROTECTION ---
                # When entering a narrow corridor, walls on the sides (angle > 60 degrees) 
                # will generate a backward radial push (rad_push < 0) that acts like a brake.
                # We cancel this backward push to let the drone slide smoothly into the gap.
                if abs(angle_lidar) > (math.pi / 3.0) and rad_push < 0.0:
                    rad_push = 0.0 

                total_rad_repulsion += rad_push
                total_orthor_repulsion += orthor_push

        # 5. SCALE DOWN
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
        :param total_correction_norm: norm of the repulsive (correction) force (given value is 0.3)
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
        repulsion_rad, repulsion_orthor = 0,0
        if is_final_approach:
            repulsion_orthor = 0.0          
            repulsion_rad = 0.5

        # Reduce speed if turning, but keep it smoother (cos^2 instead of cos^5)
        alignment_factor = max(0.2, math.cos(angle_error) ** 2)

        # 5. Active Braking & Approach
        BRAKE_DIST = 120.0 
        if dist_to_target < BRAKE_DIST:
            base_forward = max(0.15, dist_to_target * 0.03) 
            base_forward *= alignment_factor
            if self.current_speed > 4.0: base_forward = -0.4 
        else:
            base_forward = self.drone.MAX_SPEED * alignment_factor
        
        forward_cmd = base_forward + repulsion_rad

        if is_reversing: forward_cmd = -forward_cmd
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        if abs(angle_error) > 0.5 and not is_reversing:
            repulsion_orthor += -0.5 * np.sign(angle_error)

        # 7. Front-approach grasp logic during rescue
        front_grasp_cmd = self.front_grasp_alignment_command()
        if front_grasp_cmd is not None:
            #print(f'{self.drone.identifier} {self.drone.state} front_grasp_alignment_command_pilot')
            return front_grasp_cmd

        grasper_val = 1 if self.drone.grasped_wounded_persons() else 0
        #print(forward_cmd, np.clip(cmd_lateral, -1.0, 1.0), rotation_cmd, grasper_val)
        #print(f'{self.drone.identifier} {self.drone.state} move_to_target_carrot_last_pilot')
        return self.move_function(forward = forward_cmd, lateral = 0, rotation = rotation_cmd, grasper = grasper_val, repulsive_force_bool = True)