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

    def calculate_repulsive_force(self):
        """
        Only calculate the force to get the Lateral (dodge) component.
        """
        total_lat = 0.0
        #We no longer need the total_fwd, but we still include it to ensure the vortex logic works correctly.
        
        semantic_data = self.drone.semantic_values()
        if not semantic_data: return 0.0, 0.0

        drone_count_nearby = 0 
        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                dist = data.distance
                if 0.1 < dist < 120.0:
                    drone_count_nearby += 1
                    
                    # Force coefficient K. 
                    # Since there is no longer forward braking force, we need a sufficiently strong lateral force to dodge in time.                    
                    K = 400.0 
                    if self.drone.not_grapsed: K = 0
                    
                    force_magnitude = K / (dist ** 2)
                    force_magnitude = min(1.2, force_magnitude)

                    # VORTEX Logic (Xoáy)
                    VORTEX_ANGLE = 0.4 
                    if drone_count_nearby > 2: VORTEX_ANGLE = 0.8 

                    push_angle = data.angle + math.pi - VORTEX_ANGLE
                    
                    # total_fwd += ... (ignore, do not use)
                    total_lat += force_magnitude * math.sin(push_angle)
        
        return 0.0, total_lat # Returns only lateral
    
    def move_to_target_PID(self) -> CommandsDict:
        """
        Your ORIGINAL control function + additional lateral force field.
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
        MAX_SPEED = 0.5
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
            forward_cmd = 0.7
            if dist_to_target <= 60.0:
                forward_cmd = 0.45

        # Initialize default lateral command
        cmd_lateral = 0.0

        # --------------------------------------------------
        # 4. Drone collision avoidance (OLD deadlock resolution)
        # Hard safety layer to prevent drone-to-drone blocking
        # --------------------------------------------------
        if forward_cmd > 0.05 and self.is_blocked_by_drone(safety_dist=60.0):
            forward_cmd = 0.0 
            cmd_lateral = -0.6 

        # --------------------------------------------------
        # 5. [NEW INTEGRATION] Add lateral repulsive force
        # Only affects sideways motion, not forward speed
        # --------------------------------------------------
        _, rep_lat = self.calculate_repulsive_force()
        
        # Accumulate lateral forces
        cmd_lateral += rep_lat
        
        # Clamp lateral command
        cmd_lateral = max(-1.0, min(1.0, cmd_lateral))

        # --------------------------------------------------
        # Smart grasper logic
        # Automatically grab/release depending on state
        # --------------------------------------------------
        grasper_val = 0
        if self.drone.state in ["RETURNING", "DROPPING"]:
            grasper_val = 1
        elif self.drone.state == "RESCUING":
            if dist_to_target <= 15.0:
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

                        return {
                            "forward": forward_cmd,
                            "lateral": cmd_lateral,
                            "rotation": rotation_cmd, 
                            "grasper": grasper_val
                        }

        # Final command output
        return {
            "forward": forward_cmd, 
            "lateral": cmd_lateral, 
            "rotation": rotation_cmd, 
            "grasper": grasper_val
        }