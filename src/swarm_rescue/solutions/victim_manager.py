import numpy as np
import math
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class VictimManager:
    def __init__(self):
        # Danh sách nạn nhân: [{'id': str, 'pos': np.array, 'ts': int}]
        self.registry = []
        self.merge_threshold = 80.0 # Gộp nếu cách nhau < 80cm

    def update_from_sensor(self, drone_pos, drone_angle, semantic_data, current_step):
        if not semantic_data: return

        observed_victims = []
        observed_drones = []

        # 1. Phân loại dữ liệu cảm biến
        for data in semantic_data:
            # Tính tọa độ tuyệt đối (World Coordinates)
            angle_global = drone_angle + data.angle
            wx = drone_pos[0] + data.distance * math.cos(angle_global)
            wy = drone_pos[1] + data.distance * math.sin(angle_global)
            pos = np.array([wx, wy])

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                # Nếu simulator đã đánh dấu là grasped thì bỏ qua luôn
                if getattr(data, 'grasped', False): continue
                observed_victims.append(pos)
            
            elif data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                observed_drones.append(pos)

        # 2. Lọc nạn nhân đang bị Drone khác gắp (ANTI-STEAL LOGIC)
        # Ngưỡng 30cm: Nếu victim nằm quá gần drone khác -> Coi như đã bị gắp
        SAFE_DISTANCE_FROM_OTHER_DRONE = 30.0 

        for v_pos in observed_victims:
            is_being_carried = False
            for d_pos in observed_drones:
                if np.linalg.norm(v_pos - d_pos) < SAFE_DISTANCE_FROM_OTHER_DRONE:
                    is_being_carried = True
                    break
            
            if is_being_carried: continue # Bỏ qua, không lưu

            # 3. Logic Ghi Sổ (Tracking)
            found = False
            for record in self.registry:
                dist = np.linalg.norm(record['pos'] - v_pos)
                if dist < self.merge_threshold:
                    # Cập nhật vị trí mới chính xác hơn
                    record['pos'] = v_pos 
                    record['ts'] = current_step
                    found = True
                    break
            
            if not found:
                # Tạo ID tạm dựa trên tọa độ
                vid = f"{int(v_pos[0])}_{int(v_pos[1])}"
                self.registry.append({
                    'id': vid,
                    'pos': v_pos,
                    'ts': current_step
                })

    def get_nearest_victim(self, drone_pos):
        if not self.registry: return None
        closest_dist = float('inf')
        best_pos = None
        for record in self.registry:
            dist = np.linalg.norm(record['pos'] - drone_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_pos = record['pos']
        return best_pos

    def delete_victim_at(self, position, radius=100.0):
        """Xóa nạn nhân tại khu vực đã cứu xong hoặc xác định là ảo"""
        # Giữ lại những người Ở XA vị trí xóa (khoảng cách > radius)
        self.registry = [r for r in self.registry if np.linalg.norm(r['pos'] - position) > radius]