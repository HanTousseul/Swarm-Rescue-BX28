import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

# Import các module cần thiết từ framework
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

# --- CẤU HÌNH ---
SAFE_DISTANCE = 40      # Khoảng cách an toàn (pixel) để tránh va chạm
KP_ROTATION = 2.0       # Hệ số P cho xoay
KP_FORWARD = 0.5        # Hệ số P cho tiến
MAX_LIDAR_RANGE = 150   # Ngưỡng để coi là "biên giới" (frontier)
REACH_THRESHOLD = 10.0  # Khoảng cách để coi là đã đến đích

class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # --- 1. NAVIGATOR VARIABLES (Quoc Viet & Anhad) ---
        # Vị trí và góc ước tính (Tin cậy hơn GPS thô)
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None # Để kiểm tra khi nào GPS bị mất/có lại

        # --- 2. MAPPER VARIABLES (Marc) ---
        self.edge = {}
        self.visited_node = []
        
        # --- 3. COMMANDER VARIABLES (Van Khue) ---
        self.state = "EXPLORING" # EXPLORING, RESCUING, RETURNING, DROPPING
        self.path_history = {}
        self.current_target = None # Điểm đang đi tới (np.array)
        self.rescue_center_pos = None # Vị trí trạm cứu hộ (lưu khi thấy)

    def update_navigator(self):
        """Cập nhật vị trí ước tính dựa trên GPS (nếu có) hoặc Odometer."""
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            # GPS TỐT: Tin tưởng GPS
            self.estimated_pos = gps_pos
            self.estimated_angle = compass_angle
            self.gps_last_known = gps_pos
        
        else:
            # MẤT GPS: Dùng Odometer để cộng dồn (Dead Reckoning)
            odom = self.odometer_values() # [dist, alpha, theta]
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                
                # Logic của Quoc Viet:
                # Alpha là hướng di chuyển tương đối so với hướng CŨ
                move_angle = self.estimated_angle + alpha
                
                self.estimated_pos[0] += dist * math.cos(move_angle)
                self.estimated_pos[1] += dist * math.sin(move_angle)
                
                # Cập nhật góc mới
                self.estimated_angle = normalize_angle(self.estimated_angle + theta)

    def lidar_possible_paths(self) -> List:
        '''This function's purpose is to collect Lidar data, analyse it, and output a list containing potential areas to explore at that timestep. all rays from -180 to -135 and 135 to 180 degrees are discarded because, in theory, the drone is coming from behind himself, so this is not a possible path. the list of possible paths is list containing tuples of the angle to explore, along with a boolean value corresponding to wether this node has been explored or not. All booleans will be False in this functin but should be changed later (DFS algorithm) '''
        list_possible_area=[]
        #list_test=[] test list that stores the min and max rays that help populate the list_possible_area list
        min_ray=-3/4*math.pi,0
        max_ray=0,0
        ray_ini=False
        minimal_distance=285
        coords=self.gps_values()
        angle=self.measured_compass_angle()
        step_forward=50

        if not self.lidar_is_disabled():
            lidar_data=self.lidar_values()
            ray_angles = self.lidar_rays_angles()
            for i in range (22,len(lidar_data)-22):
                if lidar_data[i]>minimal_distance: 
                    if lidar_data[i-1]<=minimal_distance:
                        if i==22:
                            ray_ini=True
                        min_ray=ray_angles[i],i
                else:
                    if i!=0 and lidar_data[i-1]>minimal_distance:
                        max_ray=ray_angles[i-1],i-1
                        if max_ray!=min_ray and min_ray[1]+3<max_ray[1]:
                            #list_test.append((min_ray,max_ray))
                            list_possible_area.append(((coords[0]+step_forward*(math.cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(math.sin(angle+(min_ray[0]+max_ray[0])/2))),False))
                if i==len(lidar_data)-23 and min_ray[1]>max_ray[1]:
                    if ray_ini:
                        boolean=True

                        for k in range(min_ray[1],len(lidar_data)+22):
                            if boolean:
                                if lidar_data[i%181]<=minimal_distance:
                                    boolean=False

                        if boolean:
                            del list_possible_area[0]
                            #list_test.append((min_ray,max_ray))
                            list_possible_area.append(((coords[0]+step_forward*(math.cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(math.sin(angle+(min_ray[0]+max_ray[0])/2))),False))
                            return list_possible_area

                    max_ray=ray_angles[i],i
                    #list_test.append((min_ray,max_ray))
                    list_possible_area.append(((coords[0]+step_forward*(math.cos(angle+(min_ray[0]+max_ray[0])/2)),coords[1]+step_forward*(math.sin(angle+(min_ray[0]+max_ray[0])/2))),False))
        
        return list_possible_area

    def update_mapper(self):
        """Quét Lidar để tìm các điểm biên giới (Frontiers) mới."""
        list_possible_area = self.lidar_possible_paths()
        pos_key = tuple(self.estimated_pos)
        if pos_key not in self.edge:
            self.edge[pos_key] = [] 
        for val in list_possible_area:
            x = val[0][0]
            y = val[0][1]
            visited = False
            for node in self.visited_node:
                delta_x = x - node[0]
                delta_y = y - node[1]
                dist_to_target = math.hypot(delta_x, delta_y)
                if dist_to_target < 30: visited = True
            if not visited: 
                self.edge[pos_key].append((x,y))
                print(f'Add new target {x}, {y}')


    def move_to_target(self) -> CommandsDict:
        """Điều khiển drone đi đến self.current_target."""
        print(f'Going to {self.current_target}')
        if self.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Logic P-Controller của Anhad (đã sửa)
        delta_x = self.current_target[0] - self.estimated_pos[0]
        delta_y = self.current_target[1] - self.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # 1. Xoay về hướng mục tiêu
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.estimated_angle)
        
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # 2. Tiến tới (giảm tốc nếu cần xoay nhiều)
        forward_cmd = KP_FORWARD * dist_to_target
        if abs(angle_error) > 0.5: # Nếu góc lệch > 30 độ
            forward_cmd = 0.0      # Dừng lại để xoay cho chuẩn
        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # 3. Tránh va chạm khẩn cấp (Lidar Safety)
        lidar_vals = self.lidar_values()
        if lidar_vals is not None:
            # Kiểm tra phía trước (index 90)
            if lidar_vals[90] < SAFE_DISTANCE:
                forward_cmd = 0.0 # Dừng khẩn cấp
                rotation_cmd = 1.0 # Xoay trái để thoát

        grasper_val = 1 if (self.state == "RESCUING" or self.state == "RETURNING") else 0

        return {
            "forward": forward_cmd, 
            "lateral": 0.0, 
            "rotation": rotation_cmd, 
            "grasper": grasper_val
        }
    
    def visit(self, pos):
        if pos is not None:
            pos_key = tuple(pos) if isinstance(pos, np.ndarray) else pos
            if pos_key not in self.visited_node: 
                print(f'Add {pos_key} to visited nodes')
                self.visited_node.append(pos_key)

    def control(self) -> CommandsDict:
        
        # 1. Cập nhật Navigator (Luôn chạy đầu tiên)
        self.update_navigator()
        
        # 3. Xử lý Cảm biến Semantic (Tìm người / Trạm)
        semantic_data = self.semantic_values()
        found_person_pos = None
        found_rescue_pos = None
        
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    # Tính vị trí người dựa trên góc/khoảng cách tương đối
                    angle_global = self.estimated_angle + data.angle
                    px = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    py = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    found_person_pos = np.array([px, py])
                
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    # Lưu vị trí trạm để dùng sau này
                    if self.rescue_center_pos is None:
                        angle_global = self.estimated_angle + data.angle
                        rx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                        ry = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                        self.rescue_center_pos = np.array([rx, ry])
                    found_rescue_pos = True

        # 4. MÁY TRẠNG THÁI (STATE MACHINE)
        
        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            # Nếu thấy người -> Chuyển sang cứu
            if found_person_pos is not None:
                self.state = "RESCUING"
                self.current_target = found_person_pos
            
            # Nếu chưa có mục tiêu hoặc đã đến mục tiêu cũ
            elif self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                self.visit(self.current_target)
                print(f'Arrived {self.current_target}')
                
                # 2. Cập nhật Mapper (Để tìm đường mới)
                self.update_mapper()
                if self.current_target is None: self.visit(self.estimated_pos)

                pos_key = tuple(self.estimated_pos)
                if pos_key in self.edge and len(self.edge[pos_key]):
                    next_target = self.edge[pos_key].pop()
                    
                    if self.current_target is None: self.path_history[next_target] = self.estimated_pos
                    else: self.path_history[next_target] = self.current_target
                    
                    self.current_target = np.array(next_target)
                else:
                    # Hết đường khám phá -> Quay về node trước đó
                    current_key = tuple(self.current_target) if self.current_target is not None else None
                    if current_key and current_key in self.path_history:
                         self.current_target = self.path_history[current_key]
                    else:
                        # Xử lý khi không có đường quay về
                        print("No parent node found, staying at current position")
                        self.current_target = self.estimated_pos.copy()
                print(f'Choose next target: {self.current_target}')

        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":
            # Cập nhật vị trí người liên tục cho chính xác
            if found_person_pos is not None:
                self.current_target = found_person_pos
            
            # Kiểm tra xem đã gắp được chưa 
            if self.grasped_wounded_persons():
                self.state = "RETURNING"
                # Mục tiêu đầu tiên khi quay về là trạm cứu hộ (nếu biết) hoặc quay lại đường cũ
                if self.rescue_center_pos is not None:
                    self.current_target = self.rescue_center_pos
                else:
                    # Fallback: Quay lại vị trí trước khi cứu
                    self.current_target = np.array([0.0, 0.0]) # (Cần logic backtrack tốt hơn ở đây)

        # --- STATE: RETURNING ---
        elif self.state == "RETURNING":
            # Nếu thấy trạm cứu hộ -> đi thẳng vào đó
            if self.rescue_center_pos is not None:
                self.current_target = self.rescue_center_pos
            
            # Nếu đến nơi (trạm)
            if found_rescue_pos and np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                self.state = "DROPPING"

        # --- STATE: DROPPING ---
        elif self.state == "DROPPING":
            # Dừng và thả
            if not self.grasped_wounded_persons():
                self.state = "EXPLORING" # Quay lại tìm tiếp
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # 5. Thực hiện di chuyển
        return self.move_to_target()

    def define_message_for_all(self):
        pass