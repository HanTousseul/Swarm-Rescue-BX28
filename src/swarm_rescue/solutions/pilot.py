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
        """Lực né Drone đồng đội (Giữ nguyên)."""
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

    def calculate_wall_repulsion(self, aggressive: bool = False):
        """
        [UPDATE] Lực né Tường có chế độ Aggressive.
        Nếu aggressive=True (đang cứu người/vào ngõ hẹp), lực đẩy tường sẽ yếu đi.
        """
        lidar = self.drone.lidar_values()
        angles = self.drone.lidar_rays_angles()
        if lidar is None or angles is None: return 0.0, 1.0 

        total_lat = 0.0
        min_dist_detected = 300.0

        # [TUNE] Cấu hình lực đẩy
        if aggressive:
            # Chế độ "Lì đòn": Chỉ sợ khi quá gần (< 40cm), lực đẩy yếu (150)
            K_wall = 150.0 
            ignore_dist = 40.0 
            critical_dist = 10.0
            slow_down_threshold = 40.0
        else:
            # Chế độ "An toàn": Sợ từ xa (< 80cm), lực đẩy mạnh (400)
            K_wall = 400.0
            ignore_dist = 80.0
            critical_dist = 20.0
            slow_down_threshold = 60.0

        step = 5
        for i in range(0, len(lidar), step):
            dist = lidar[i]
            if 10.0 < dist < ignore_dist:
                if dist < min_dist_detected: min_dist_detected = dist
                
                # Tính lực đẩy
                force = K_wall / (dist ** 1.8)
                force = min(1.0, force) 

                angle_obs = angles[i]
                push_angle = angle_obs + math.pi 
                total_lat += force * math.sin(push_angle)

        # Tính hệ số giảm tốc
        # Nếu đang aggressive, cho phép giữ tốc độ cao hơn khi gần tường
        speed_factor = np.clip((min_dist_detected - critical_dist) / slow_down_threshold, 0.3, 1.0)
        
        # Nếu aggressive, speed factor tối thiểu là 0.5 (để đủ đà lao vào)
        if aggressive:
            speed_factor = max(0.5, speed_factor)

        return total_lat, speed_factor

    def move_to_target_carrot(self) -> CommandsDict:
        if self.drone.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        if self.last_pos is not None:
            move_dist = np.linalg.norm(self.drone.estimated_pos - self.last_pos)
            self.current_speed = move_dist 
        self.last_pos = self.drone.estimated_pos.copy()

        delta_x = self.drone.current_target[0] - self.drone.estimated_pos[0]
        delta_y = self.drone.current_target[1] - self.drone.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)
        target_angle = math.atan2(delta_y, delta_x)

        # Xác định chế độ bay
        is_reversing = (self.drone.grasped_wounded_persons() and self.drone.state == 'RETURNING')
        
        # [NEW] Chế độ AGGRESSIVE (Quyết tâm)
        # Bật khi: Đang cứu người HOẶC Đích đến rất gần (< 80cm - tức là đang chui hẹp)
        is_aggressive = (self.drone.state == 'RESCUING') or (dist_to_target < 80.0)

        if is_reversing:
            desired_angle = normalize_angle(target_angle + math.pi)
        else:
            desired_angle = target_angle

        angle_error = normalize_angle(desired_angle - self.drone.estimated_angle)
        
        KP_ROT = 5.0 
        rotation_cmd = KP_ROT * angle_error
        rotation_cmd = np.clip(rotation_cmd, -1.0, 1.0)

        # [UPDATE] Truyền cờ aggressive vào hàm tính tường
        wall_lat, wall_speed_factor = self.calculate_wall_repulsion(aggressive=is_aggressive)

        MAX_SPEED = 1.2 
        cornering_factor = max(0.1, math.cos(angle_error) ** 2)
        forward_cmd = MAX_SPEED * cornering_factor * wall_speed_factor

        # Active Braking
        BRAKE_DIST = 50.0 
        if dist_to_target < BRAKE_DIST:
            forward_cmd = dist_to_target * 0.04 
            if self.current_speed > 5.0: 
                forward_cmd = -0.6 

        if is_reversing: forward_cmd = -forward_cmd
        forward_cmd = np.clip(forward_cmd, -1.0, 1.0)

        # Lateral Control
        cmd_lateral = 0.0
        _, drone_lat = self.calculate_repulsive_force()
        
        cmd_lateral = drone_lat + wall_lat

        if abs(angle_error) > 0.5 and not is_reversing:
            cmd_lateral += -0.5 * np.sign(angle_error)

        grasper_val = 1 if (self.drone.grasped_wounded_persons() or (self.drone.state == "RESCUING" and dist_to_target < 40.0)) else 0

        return {
            "forward": forward_cmd,
            "lateral": np.clip(cmd_lateral, -1.0, 1.0),
            "rotation": rotation_cmd,
            "grasper": grasper_val
        }