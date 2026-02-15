import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.comm_counter = 0 

    # [NEW] Gửi thông báo XÍ CHỖ
    def broadcast_claim_target(self, target_pos: np.ndarray):
        if self.drone.communicator_is_disabled(): return
        
        msg = {
            "id": self.drone.identifier,
            "type": "CLAIM_TARGET",
            "pos": target_pos
        }
        self.drone.outgoing_msg_buffer = msg

    # [NEW] Gửi bản đồ vật cản (Chia sẻ tầm nhìn)
    def broadcast_obstacle_update(self):
        if self.drone.communicator_is_disabled(): return

        # Chỉ gửi mỗi 20 nhịp (0.3s) để tránh spam nghẽn mạng
        self.comm_counter += 1
        if self.comm_counter % 20 != 0: return

        lidar_vals = self.drone.lidar_values()
        lidar_angles = self.drone.lidar_rays_angles()
        if lidar_vals is None: return

        # Nén dữ liệu: Chỉ lấy các điểm tường rõ ràng (< 3m) và lấy mẫu thưa
        obstacle_points = []
        step = 10 
        for i in range(0, len(lidar_vals), step):
            dist = lidar_vals[i]
            if 0.1 < dist < 290.0: 
                angle = self.drone.estimated_angle + lidar_angles[i]
                ox = self.drone.estimated_pos[0] + dist * np.cos(angle)
                oy = self.drone.estimated_pos[1] + dist * np.sin(angle)
                obstacle_points.append((round(ox, 1), round(oy, 1)))

        if obstacle_points:
            msg = {
                "id": self.drone.identifier,
                "type": "MAP_OBSTACLE",
                "points": obstacle_points
            }
            self.drone.outgoing_msg_buffer = msg

    def broadcast_map_updates(self):
        """Gửi vị trí nạn nhân (Heatmap update)"""
        if self.drone.communicator_is_disabled(): return
        
        # Logic cũ: thỉnh thoảng gửi vị trí victim tốt nhất
        if self.comm_counter % 15 != 0: return

        best_target = self.drone.nav.victim_map.get_highest_score_target()
        if best_target is not None:
            msg = {
                "id": self.drone.identifier,
                "type": "MAP_UPDATE",
                "victim_found": True,
                "victim_pos": best_target
            }
            self.drone.outgoing_msg_buffer = msg

    def broadcast_clear_zone(self, position: np.ndarray):
        if self.drone.communicator_is_disabled(): return
        msg = {
            "id": self.drone.identifier,
            "type": "CLEAR_ZONE", 
            "pos": position
        }
        self.drone.outgoing_msg_buffer = msg

    def process_incoming_messages(self):
        if self.drone.communicator_is_disabled(): return

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            sender_id = content.get("id")
            if sender_id == self.drone.identifier: continue 

            msg_type = content.get("type")
            
            # 1. Nhận tin XÍ CHỖ
            if msg_type == "CLAIM_TARGET":
                pos = content.get("pos")
                if pos is not None:
                    # Lưu vào danh sách bận của Driver (Hiệu lực 6-7 giây)
                    self.drone.busy_targets.append({
                        "pos": np.array(pos),
                        "timer": 400 
                    })

            # 2. Nhận tin OBSTACLE (Vẽ lên bản đồ mình)
            elif msg_type == "MAP_OBSTACLE":
                points = content.get("points")
                if points:
                    self.drone.nav.obstacle_map.update_from_remote_points(points)

            # 3. Nhận tin CLEAR ZONE
            elif msg_type == "CLEAR_ZONE":
                clear_pos = content.get("pos")
                if clear_pos is not None:
                    # Gọi hàm clear_area với bán kính lớn (5)
                    self.drone.nav.victim_map.clear_area(np.array(clear_pos), radius_grid=5)
            
            # 4. Nhận tin VICTIM
            elif msg_type == "MAP_UPDATE" and content.get("victim_found"):
                victim_pos = content.get("victim_pos")
                if victim_pos is not None:
                    gx, gy = self.drone.nav.victim_map.world_to_grid(victim_pos[0], victim_pos[1])
                    self.drone.nav.victim_map.grid[gy, gx] += 5.0