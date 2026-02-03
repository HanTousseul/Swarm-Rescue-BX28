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
        Chỉ tính toán lực để lấy component Lateral (né tránh).
        """
        total_lat = 0.0
        # total_fwd ta không cần dùng nữa, nhưng vẫn tính để logic vortex hoạt động đúng
        
        semantic_data = self.drone.semantic_values()
        if not semantic_data: return 0.0, 0.0

        drone_count_nearby = 0 
        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                dist = data.distance
                if 0.1 < dist < 120.0:
                    drone_count_nearby += 1
                    
                    # Hệ số lực K. 
                    # Vì không còn lực hãm forward, ta cần lateral đủ mạnh để né kịp.
                    K = 400.0 
                    if self.drone.not_grapsed: K = 0
                    
                    force_magnitude = K / (dist ** 2)
                    force_magnitude = min(1.2, force_magnitude)

                    # VORTEX Logic (Xoáy)
                    VORTEX_ANGLE = 0.4 
                    if drone_count_nearby > 2: VORTEX_ANGLE = 0.8 

                    push_angle = data.angle + math.pi - VORTEX_ANGLE
                    
                    # total_fwd += ... (Bỏ qua, không dùng)
                    total_lat += force_magnitude * math.sin(push_angle)
        
        return 0.0, total_lat # Chỉ trả về lateral
    
    def move_to_target_PID(self) -> CommandsDict:
        """
        Hàm điều khiển GỐC của bạn + Cộng thêm Lateral từ trường.
        """
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # 1. Xoay về hướng mục tiêu
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.drone.estimated_angle)
        
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # 2. Tiến tới (LOGIC CŨ: CHẬM & CHẮC)
        MAX_SPEED = 0.6
        BRAKING_DIST = 150.0
        STOP_DIST = 15.0 

        if dist_to_target > BRAKING_DIST:
            forward_cmd = MAX_SPEED
        elif dist_to_target > STOP_DIST:
            forward_cmd = (dist_to_target / BRAKING_DIST) * MAX_SPEED
            forward_cmd = max(0.1, forward_cmd)
        else:
            forward_cmd = 0.05

        # 3. Kỷ luật Xoay
        if abs(angle_error) > 0.2:
            forward_cmd = 0.0 

        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # --- LOGIC ĐẶC BIỆT CHO RETURNING (Cõng người) ---
        if self.drone.grasped_wounded_persons():
            forward_cmd = 0.8
            if dist_to_target <= 60.0: forward_cmd = 0.45

        # Khởi tạo cmd_lateral mặc định
        cmd_lateral = 0.0

        # --- 4. XỬ LÝ TRÁNH VA CHẠM DRONE (DEADLOCK RESOLUTION CŨ) ---
        # Giữ lại cái này như một lớp bảo vệ cứng (Hard Safety)
        if forward_cmd > 0.05 and self.is_blocked_by_drone(safety_dist=60.0):
            forward_cmd = 0.0 
            cmd_lateral = -0.6 

        # --- [TÍCH HỢP MỚI] 5. CỘNG HƯỞNG LỰC ĐẨY LATERAL ---
        # Chỉ cộng thêm lực trượt ngang, không ảnh hưởng forward
        _, rep_lat = self.calculate_repulsive_force()
        
        # Cộng dồn vào lateral hiện tại
        cmd_lateral += rep_lat
        
        # Clip lateral để không quá giới hạn
        cmd_lateral = max(-1.0, min(1.0, cmd_lateral))

        # --- LOGIC GRASPER THÔNG MINH ---
        grasper_val = 0
        if self.drone.state in ["RETURNING", "DROPPING"]:
            grasper_val = 1
        elif self.drone.state == "RESCUING":
            if dist_to_target <= 17.0: grasper_val = 1
            else: grasper_val = 0
        elif self.drone.state == "END_GAME" and self.drone.grasped_wounded_persons():
             grasper_val = 1

        if self.drone.not_grapsed: grasper_val = 0

        # --- ANTI-STUCK MECHANISM (Rescue Center Wall) ---
        if self.drone.state in ["RETURNING", "END_GAME"] and dist_to_target < 100.0 and dist_to_target > 30.0:
            lidar_vals = self.drone.lidar_values()
            if lidar_vals is not None:
                front_rays = lidar_vals[85:95] 
                if len(front_rays) > 0:
                    min_front_dist = np.min(front_rays)
                    if min_front_dist < 15.0:
                        forward_cmd = 0.0
                        slide_force = 0.6 
                        
                        if abs(angle_error) < 0.1: cmd_lateral = -slide_force 
                        elif angle_error > 0: cmd_lateral = slide_force 
                        else: cmd_lateral = -slide_force 

                        return {
                            "forward": forward_cmd,
                            "lateral": cmd_lateral,
                            "rotation": rotation_cmd, 
                            "grasper": grasper_val
                        }

        return {
            "forward": forward_cmd, 
            "lateral": cmd_lateral, 
            "rotation": rotation_cmd, 
            "grasper": grasper_val
        }