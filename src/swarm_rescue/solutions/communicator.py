import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.comm_counter = 0 # [NEW] Bộ đếm nhịp

    def broadcast_map_updates(self):
        """
        Send significant map info to other drones.
        To save bandwidth, we mainly focus on VICTIM LOCATIONS (Heatmap).
        """
        if self.drone.communicator_is_disabled(): return

        # [NEW] CHỐNG SPAM: Chỉ gửi update mỗi 15 ticks (khoảng 0.5 - 1 giây)
        self.comm_counter += 1
        if self.comm_counter % 15 != 0:
            return

        # Get hottest target from my map
        best_target = self.drone.nav.victim_map.get_highest_score_target()
        
        if best_target is not None:
            # Create a simplified message
            msg = {
                "id": self.drone.identifier,
                "type": "MAP_UPDATE",
                "victim_found": True,
                "victim_pos": best_target
            }
            # We don't have a direct 'broadcast' method in the snippet, 
            # assuming define_message_for_all handles the outgoing data.
            # We will store this in a variable to be picked up by define_message_for_all
            self.drone.outgoing_msg_buffer = msg

    # [NEW] Hàm gửi tín hiệu đã cứu xong
    def broadcast_clear_zone(self, position: np.ndarray):
        if self.drone.communicator_is_disabled(): return
        
        # [LOG]
        print(f"[{self.drone.identifier}] 📡 BROADCAST: CLEAR ZONE at {position}")
        
        msg = {
            "id": self.drone.identifier,
            "type": "CLEAR_ZONE", 
            "pos": position
        }
        self.drone.outgoing_msg_buffer = msg

    def process_incoming_messages(self):
        """
        Read messages from others and update LOCAL MAPS.
        """
        if self.drone.communicator_is_disabled(): return

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            msg_type = content.get("type")
            
            # --- SYNC VICTIM MAP ---
            if msg_type == "MAP_UPDATE" and content.get("victim_found"):
                victim_pos = content.get("victim_pos")
                if victim_pos is not None:
                    # [LOG] Chỉ log nếu điểm này MỚI với mình (để đỡ spam)
                    if not self.drone.nav.victim_map.is_hot_spot(victim_pos):
                         print(f"[{self.drone.identifier}] 📩 RECEIVED VICTIM INFO at {victim_pos}")

                    gx, gy = self.drone.nav.victim_map.world_to_grid(victim_pos[0], victim_pos[1])
                    score_add = 5.0
                    self.drone.nav.victim_map.grid[gy, gx] += score_add
            
            # --- XỬ LÝ CLEAR ZONE ---
            elif msg_type == "CLEAR_ZONE":
                clear_pos = content.get("pos")
                if clear_pos is not None:
                    print(f"[{self.drone.identifier}] 🧹 RECEIVED CLEAR ZONE at {clear_pos}")
                    # Chuyển đổi sang tọa độ Grid
                    gx, gy = self.drone.nav.victim_map.world_to_grid(clear_pos[0], clear_pos[1])
                    
                    # Xóa sổ vùng đó trên bản đồ của MÌNH (Set về 0)
                    # Xóa vùng 5x5 ô xung quanh để chắc chắn sạch sẽ
                    range_clear = 2
                    y_min = max(0, gy - range_clear)
                    y_max = min(self.drone.nav.victim_map.grid_h, gy + range_clear + 1)
                    x_min = max(0, gx - range_clear)
                    x_max = min(self.drone.nav.victim_map.grid_w, gx + range_clear + 1)
                    
                    self.drone.nav.victim_map.grid[y_min:y_max, x_min:x_max] = 0.0
                    # print(f"Drone {self.drone.identifier}: Cleared zone at {clear_pos}")