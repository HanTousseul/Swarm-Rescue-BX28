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
        Modified control function: Allows BACKWARD flight (Reversing) when RETURNING.
        """

        # If there is no target, stop completely
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Compute vector to target
        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # =========================================================
        # [NEW LOGIC] REVERSE FLIGHT MODE (BAY LÙI)
        # =========================================================
        # Kiểm tra xem có nên bay lùi không?
        # Chỉ bay lùi khi đang RETURNING (đang kéo nạn nhân về)
        is_reversing = (self.drone.grasped_wounded_persons() and self.drone.state == 'RETURNING')

        # --------------------------------------------------
        # 1. Rotate control
        # --------------------------------------------------
        target_angle = math.atan2(delta_y, delta_x)
        
        if is_reversing:
            # Nếu đang lùi: Hướng mũi drone NGƯỢC lại với hướng đích (quay lưng về đích)
            desired_angle = normalize_angle(target_angle + math.pi)
        else:
            # Bình thường: Hướng mũi về đích
            desired_angle = target_angle

        angle_error = normalize_angle(desired_angle - self.drone.estimated_angle)
        
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # --------------------------------------------------
        # 2. Forward movement (Speed Control)
        # --------------------------------------------------
        MAX_SPEED = 0.6
        BRAKING_DIST = 150.0
        STOP_DIST = 15.0 

        # Tính độ lớn vận tốc (Speed Magnitude) trước
        if dist_to_target > BRAKING_DIST:
            speed_mag = MAX_SPEED
        elif dist_to_target > STOP_DIST:
            speed_mag = (dist_to_target / BRAKING_DIST) * MAX_SPEED
            speed_mag = max(0.15, speed_mag)
        else:
            speed_mag = 0.1

        # Nếu đang cứu người (RETURNING), ưu tiên tốc độ cao hơn
        if self.drone.grasped_wounded_persons():
            speed_mag = 0.7
            if dist_to_target <= 45.0:
                speed_mag = 0.55

        # Gán dấu cho vận tốc: Tiến (+) hoặc Lùi (-)
        if is_reversing:
            forward_cmd = -speed_mag # Số âm để bay lùi
        else:
            forward_cmd = speed_mag

        # --------------------------------------------------
        # 3. Rotation discipline
        # --------------------------------------------------
        # Nếu chưa quay đúng hướng thì chưa di chuyển vội (tránh trượt)
        if abs(angle_error) > 0.3: # Nới lỏng một chút lên 0.3
            forward_cmd = 0.0 

        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # Initialize default lateral command
        cmd_lateral = 0.0

        # --------------------------------------------------
        # 4. Drone collision avoidance
        # --------------------------------------------------
        # Lưu ý: Khi bay lùi, forward_cmd là âm, nên check abs() hoặc check < -0.05
        moving_fast = abs(forward_cmd) > 0.05
        
        if moving_fast and self.is_blocked_by_drone(safety_dist=60.0):
            forward_cmd = 0.0 
            cmd_lateral = -0.6 

        # # --------------------------------------------------
        # # 5. Lateral repulsive force
        # # --------------------------------------------------
        # _, rep_lat = self.calculate_repulsive_force()
        
        # # Lực đẩy tường/drone vẫn tác dụng đúng theo hướng trái/phải của thân drone
        # # nên không cần đổi dấu cmd_lateral
        # cmd_lateral += rep_lat
        # cmd_lateral = max(-1.0, min(1.0, cmd_lateral))

        # --------------------------------------------------
        # Smart grasper logic
        # --------------------------------------------------
        grasper_val = 0
        if self.drone.state in ["RETURNING", "DROPPING"]:
            grasper_val = 1
        elif self.drone.state == "RESCUING":
            if dist_to_target <= 20.0:
                grasper_val = 1
            else:
                grasper_val = 0
        elif self.drone.state == "END_GAME" and self.drone.grasped_wounded_persons():
             grasper_val = 1

        if self.drone.not_grapsed:
            grasper_val = 0

        # --------------------------------------------------
        # Anti-stuck mechanism
        # --------------------------------------------------
        if self.drone.state in ["RETURNING", "END_GAME"] and dist_to_target < 100.0 and dist_to_target > 30.0:
            lidar_vals = self.drone.lidar_values()
            if lidar_vals is not None:
                # [LƯU Ý QUAN TRỌNG KHI BAY LÙI]
                # Khi bay lùi, hướng di chuyển là phía SAU.
                # Sensor phía trước (front_rays) sẽ không phát hiện được tường phía sau lưng.
                # Tuy nhiên, ta vẫn giữ logic này để tránh va mũi vào chướng ngại vật khi đang xoay sở.
                
                front_rays = lidar_vals[85:95] 
                if len(front_rays) > 0:
                    min_front_dist = np.min(front_rays)
                    if min_front_dist < 15.0:
                        forward_cmd = 0.0 # Dừng lại
                        
                        # Trượt ngang để né
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