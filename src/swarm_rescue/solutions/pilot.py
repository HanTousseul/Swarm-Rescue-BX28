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
        side_sign = 1.0 if (self.drone.identifier is None or self.drone.identifier % 2 == 0) else -1.0

        align_threshold = math.radians(12.0)
        grasp_distance = 34.0

        # Overtake mode: if we are almost perfectly behind a moving victim, push harder
        # and add a tiny lateral bias so we do not stay in the exact same lane forever.
        if abs_error < math.radians(11.0) and wounded.distance > 24.0:
            overtake_speed = float(np.clip(0.74 + 0.005 * wounded.distance, 0.74, 1.0))
            return {
                "forward": overtake_speed,
                "lateral": float(0.24 * side_sign),
                "rotation": float(np.clip(2.8 * angle_error, -0.9, 0.9)),
                "grasper": 0
            }

        # Keep moving while aligning so we do not trail behind moving victims.
        if abs_error > align_threshold:
            moving_align_speed = float(np.clip(0.44 + 0.006 * wounded.distance, 0.44, 0.92))
            return {
                "forward": moving_align_speed,
                "lateral": 0.0,
                "rotation": float(np.clip(3.4 * angle_error, -1.0, 1.0)),
                "grasper": 0
            }

        if wounded.distance > grasp_distance:
            approach_speed = float(np.clip(0.52 + 0.009 * (wounded.distance - grasp_distance), 0.52, 1.0))
            return {
                "forward": approach_speed,
                "lateral": 0.0,
                "rotation": float(np.clip(2.2 * angle_error, -0.75, 0.75)),
                "grasper": 0
            }

        return {
            "forward": 0.14,
            "lateral": 0.0,
            "rotation": float(np.clip(2.0 * angle_error, -0.7, 0.7)),
            "grasper": 1
        }

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

        for elt in range (len(lidar_data)):

            if lidar_data[elt] < 120:

                WALL_CONSTANT = 300 if self.drone.state == 'DISPERSING' else 5
                force = WALL_CONSTANT / lidar_data[elt] ** 2 
                unit_vector_angle = ray_angles[elt] + math.pi

                total_rad_repulsion += force * np.cos(unit_vector_angle)
                total_orthor_repulsion += force *np.sin(unit_vector_angle)

        # 2. SEMANTIC ENTITIES AVOIDANCE & ATTRACTION
        if semantic_data is not None:
            for elt in semantic_data:
                
                # A. ATTRACTION LOGIC: Pull towards wounded (Rescuing) or Base (Returning)
                if (elt.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and self.drone.state == 'RESCUING') or \
                   (self.drone.state == 'RETURNING' and elt.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    force = 0.1 # 1/10
                    total_rad_repulsion += force * np.cos(elt.angle)
                    total_orthor_repulsion += force * np.sin(elt.angle)
                
                # B. [NEW] REPULSION LOGIC: Push away from Rescue Center during warmup
                elif self.drone.state == 'DISPERSING' and elt.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    dist = max(elt.distance, 1.0)
                    force = 3000.0 / (dist ** 2) # Strong repulsive force to clear the base
                    push_angle = elt.angle + math.pi # Point in the opposite direction
                    total_rad_repulsion += force * np.cos(push_angle)
                    total_orthor_repulsion += force * np.sin(push_angle)

        #total_orthor_repulsion = min(0.7, total_orthor_repulsion)
        #total_rad_repulsion = min(0.7, total_rad_repulsion)

        # DRONE-TO-DRONE AVOIDANCE & PRIORITY YIELDING
        my_pos = self.drone.estimated_pos
        my_angle = self.drone.estimated_angle
        am_i_vip = self.drone.grasped_wounded_persons() # Am I carrying a victim?

        vip_relative = []     # List of (distance, angle) of VIP drones
        normal_relative = []  # List of (distance, angle) of Normal drones

        # A. Gather from Semantic Sensor (Highest Precision, Range: 200px)
        if semantic_data is not None:
            temp_drones = []
            temp_grasped_persons = []
            
            # 1. Classify data
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    temp_drones.append(data)
                elif data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and getattr(data, 'grasped', False):
                    temp_grasped_persons.append(data)
            
            # 2. Filter and match
            for drone_data in temp_drones:
                is_vip_drone = False
                
                # Compare drone with victim grasped
                for person_data in temp_grasped_persons:
                    dist_diff = abs(drone_data.distance - person_data.distance)
                    angle_diff = abs(normalize_angle(drone_data.angle - person_data.angle))
                    
                    if dist_diff < 30.0 and angle_diff < 0.2:
                        is_vip_drone = True
                        break
                
                if is_vip_drone:
                    vip_relative.append((drone_data.distance, drone_data.angle))
                else:
                    normal_relative.append((drone_data.distance, drone_data.angle))

        # B. Gather from Communicator (Long Range: 250px, Fallback)
        if hasattr(self.drone.comms, 'list_vip_drones'):
            for v_pos in self.drone.comms.list_vip_drones:
                dist = math.hypot(v_pos[0] - my_pos[0], v_pos[1] - my_pos[1])
                # Only use comms data if they are outside semantic range to avoid double-counting
                if dist > 180.0 and dist < 250.0: 
                    angle_global = math.atan2(v_pos[1] - my_pos[1], v_pos[0] - my_pos[0])
                    angle_local = normalize_angle(angle_global - my_angle)
                    vip_relative.append((dist, angle_local))
            
            for n_pos in self.drone.comms.list_nearby_drones:
                dist = math.hypot(n_pos[0] - my_pos[0], n_pos[1] - my_pos[1])
                if dist > 180.0 and dist < 250.0:
                    angle_global = math.atan2(n_pos[1] - my_pos[1], n_pos[0] - my_pos[0])
                    angle_local = normalize_angle(angle_global - my_angle)
                    normal_relative.append((dist, angle_local))

        # C. Apply Forces based on Social Hierarchy
        K_DRONE = 1000 if self.drone.state == 'DISPERSING' else 400 # Base repulsion constant
        # if self.drone.state == 'DISPERSING': print(K_DRONE)
        
        # Normal Drones pushing us
        if not am_i_vip: 
            # VIP ignores normal drones completely, straight path home!
            for dist, angle in normal_relative:
                if dist < 200.0: 
                    dist = max(dist, 1)
                    force = K_DRONE / (dist ** 2)
                    push_angle = normalize_angle(angle + math.pi)
                    if abs(angle) < 0.3 or abs(normalize_angle(angle - math.pi)) < 0.3:
                        push_angle += 0.5
                    total_rad_repulsion += force * np.cos(push_angle)
                    total_orthor_repulsion += force * np.sin(push_angle)

        # VIP Drones pushing us
        for dist, angle in vip_relative:
            if dist < 250.0:
                if am_i_vip:
                    # VIP vs VIP: Normal symmetric push to resolve deadlocks
                    dist = max(dist, 1)
                    force = K_DRONE / (dist ** 2)
                else:
                    # Normal vs VIP: HUGE PUSH. Normal drone yields heavily.
                    dist = max(dist, 1)
                    force = (K_DRONE * 5.0) / (dist ** 2)
                    # Add a slight backward brake to yield effectively
                    total_rad_repulsion -= 0.5 * force 
                
                push_angle = normalize_angle(angle + math.pi)
                if abs(angle) < 0.3 or abs(normalize_angle(angle - math.pi)) < 0.3:
                        push_angle += 0.5
                total_rad_repulsion += force * np.cos(push_angle)
                total_orthor_repulsion += force * np.sin(push_angle)

        actual_norm_correction = math.hypot(total_rad_repulsion,total_orthor_repulsion)
        if actual_norm_correction < 0.001: return 0,0

        MAX_REPULSION = 2.5
        if actual_norm_correction > MAX_REPULSION:
            total_rad_repulsion *= MAX_REPULSION / actual_norm_correction
            total_orthor_repulsion *= MAX_REPULSION / actual_norm_correction

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

        if abs(angle_error) > 0.5 and not is_reversing:
            repulsion_orthor += -0.5 * np.sign(angle_error)

        if is_final_approach:
            repulsion_orthor = 0.0          
            repulsion_rad = 0.5

        # Reduce speed if turning, but keep it smoother (cos^2 instead of cos^5)
        alignment_factor = max(0.2, math.cos(angle_error) ** 2)

        # 5. Active Braking & Approach
        BRAKE_DIST = 120.0 
        if dist_to_target < BRAKE_DIST:
            # [FIXED] Tăng mức tối thiểu từ 0.15 lên 0.4 để có đủ lực "ủi" vào bãi đỗ
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

        # --- [NEW] CHỈ GIẢM XÓC TAY LÁI, KHÔNG HÃM TỐC ĐỘ ---
        # Giúp lướt mượt qua hành lang nhưng vẫn giữ nguyên đà tiến
        lidar_values = self.drone.lidar_values()
        if lidar_values is not None and min(lidar_values) < 60.0:
            if abs(angle_error) < 0.5:
                rotation_cmd *= 0.5

        # --- Fix error adding wrong force when go back ---
        if not is_reversing:
            forward_cmd = base_forward + repulsion_rad
        else:
            forward_cmd = -base_forward + repulsion_rad
            
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        # 7. Front-approach grasp logic during rescue
        front_grasp_cmd = self.front_grasp_alignment_command()
        if front_grasp_cmd is not None:
            #print(f'{self.drone.identifier} {self.drone.state}front_grasp_alignment_command_pilot')
            return front_grasp_cmd

        grasper_val = 1 if self.drone.grasped_wounded_persons() else 0
        ## print(forward_cmd, np.clip(cmd_lateral, -1.0, 1.0), rotation_cmd, grasper_val)

        return {
            "forward": float(forward_cmd),
            "lateral": float(np.clip(repulsion_orthor, -1.0, 1.0)),
            "rotation": float(rotation_cmd),
            "grasper": int(grasper_val)
        }
