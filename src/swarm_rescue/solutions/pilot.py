import math
import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class Pilot:
    def __init__(self, drone):
        self.drone = drone
        self.last_pos = None
        self.current_speed = 0.0

    def calculate_repulsive_force(self):
        """Lực né Drone đồng đội."""
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

    def calculate_wall_repulsion(self, aggressive: bool = False, angle_error: float = 0.0): # [UPDATE] Thêm tham số angle_error
        """Lực né Tường."""
        lidar = self.drone.lidar_values()
        angles = self.drone.lidar_rays_angles()
        if lidar is None or angles is None: return 0.0, 1.0 

        total_lat = 0.0
        min_dist_detected = 300.0

        if aggressive:
            K_wall = 150.0; ignore_dist = 40.0; critical_dist = 10.0; slow_down_threshold = 40.0
        else:
            K_wall = 400.0; ignore_dist = 80.0; critical_dist = 20.0; slow_down_threshold = 60.0

        # [NEW] Logic giảm lực đẩy khi đang đi đúng hướng (Lách qua cửa)
        # Nếu góc lệch nhỏ (< 10 độ), chứng tỏ ta đang chủ đích lao vào khe đó
        # -> Giảm K_wall đi một nửa để không bị bật ra
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

    def move_to_target_carrot(self) -> CommandsDict:
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

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
        
        # Final Approach: Cực gần (< 25cm) và đang đi cứu
        is_final_approach = (self.drone.state == 'RESCUING' and dist_to_target < 25.0)
        is_aggressive = (self.drone.state == 'RESCUING') or (dist_to_target < 80.0)

        if is_reversing:
            desired_angle = normalize_angle(target_angle + math.pi)
        else:
            desired_angle = target_angle

        angle_error = normalize_angle(desired_angle - self.drone.estimated_angle)
        
        # [TUNE] Giảm KP xuống để bớt lắc
        if is_final_approach:
            KP_ROT = 4.0 # Gần đích thì xoay mạnh hơn chút để align
        else:
            KP_ROT = 2.5 # Bình thường xoay từ tốn thôi

        rotation_cmd = KP_ROT * angle_error
        rotation_cmd = np.clip(rotation_cmd, -1.0, 1.0)

        wall_lat, wall_speed_factor = self.calculate_wall_repulsion(aggressive=is_aggressive, angle_error=angle_error)

        if is_final_approach:
            wall_lat = 0.0 
            wall_speed_factor = 1.0 

        MAX_SPEED = 1.2 
        
        # [NEW] ALIGN THEN MOVE: Nếu góc lệch lớn, giảm tốc độ tiến
        # cos^5 phạt rất nặng nếu góc lệch > 30 độ -> Drone sẽ tự chậm lại để xoay
        alignment_factor = max(0.0, math.cos(angle_error) ** 5)
        
        forward_cmd = MAX_SPEED * alignment_factor * wall_speed_factor

        # Active Braking
        BRAKE_DIST = 60.0 
        if dist_to_target < BRAKE_DIST:
            forward_cmd = max(0.15, dist_to_target * 0.03) # Giữ min speed 0.15 để không dừng hẳn
            if self.current_speed > 5.0: forward_cmd = -0.5 

        if is_reversing: forward_cmd = -forward_cmd
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        cmd_lateral = 0.0
        _, drone_lat = self.calculate_repulsive_force()
        cmd_lateral = drone_lat + wall_lat

        if abs(angle_error) > 0.5 and not is_reversing:
            cmd_lateral += -0.5 * np.sign(angle_error) # Drift assist

        # [FIX] FRONT GRASP ONLY: Chỉ gắp khi mặt hướng về nạn nhân
        # abs(angle_error) < 1.0 (khoảng 60 độ) và khoảng cách < 25cm
        can_grasp = (dist_to_target < 20.0) and (abs(angle_error) < 1.0)
        
        grasper_val = 1 if (self.drone.grasped_wounded_persons() or (self.drone.state == "RESCUING" and can_grasp)) else 0

        return {
            "forward": forward_cmd,
            "lateral": np.clip(cmd_lateral, -1.0, 1.0),
            "rotation": rotation_cmd,
            "grasper": grasper_val
        }