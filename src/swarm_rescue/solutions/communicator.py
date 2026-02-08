import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone

    # Đã xóa hàm should_wait_in_queue() để loại bỏ cơ chế xếp hàng

    def is_target_taken_or_better_candidate(self, target_person_pos):
        if target_person_pos is None: return False
        if self.drone.communicator_is_disabled(): return False
        
        my_dist = np.linalg.norm(self.drone.estimated_pos - target_person_pos)
        COORDINATE_MATCH_THRESHOLD = 50.0 

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            other_id = content.get("id")
            other_state = content.get("state")
            other_person_pos = content.get("person_pos")
            other_current_pos = content.get("current_pos")
            
            if other_person_pos is None or other_current_pos is None: continue
            
            dist_between_targets = np.linalg.norm(target_person_pos - other_person_pos)
            
            # Nếu cùng nhắm vào một người
            if dist_between_targets < COORDINATE_MATCH_THRESHOLD:
                
                other_dist_to_person = np.linalg.norm(other_current_pos - other_person_pos)

                # --- CASE 1: ĐỐI THỦ ĐÃ CHỐT KÈO (RETURNING/DROPPING) ---
                if other_state in ["RETURNING", "DROPPING"]:
                    return True 

                # --- CASE 2: CẢ HAI CÙNG TRANH NHAU (RESCUING vs RESCUING) ---
                # Hoặc EXPLORING vs RESCUING, EXPLORING vs EXPLORING
                # Logic: So sánh khoảng cách và ID
                
                # Nếu đối thủ gần hơn mình đáng kể (> 20px) -> Nhường
                if other_dist_to_person < my_dist - 20.0:
                    return True
                
                # Nếu khoảng cách ngang nhau (trong phạm vi 20px) -> So ID
                # Ai có ID nhỏ hơn thì được quyền ưu tiên (Luật bất thành văn để phá thế kẹt)
                if abs(other_dist_to_person - my_dist) <= 20.0:
                    if other_id < self.drone.identifier:
                        return True

        return False