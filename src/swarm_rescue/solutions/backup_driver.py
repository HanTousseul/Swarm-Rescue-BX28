import math
import numpy as np
import os
# from stable_baselines3 import PPO
from typing import Optional, List, Tuple, Dict, Any

# Import necessary modules from framework
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


# --- CONFIGURATION ---
SAFE_DISTANCE = 40      # Safe distance (pixels) to avoid collisions
KP_ROTATION = 2.0       # P coefficient for rotation
KP_FORWARD = 0.5        # P coefficient for forward movement
MAX_LIDAR_RANGE = 150   # Threshold to consider as "frontier"
REACH_THRESHOLD = 25.0  # Distance to consider as reached destination
STEPS_TO_RETURN = 300


class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # --- 1. NAVIGATOR VARIABLES ---
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None 

        # --- 2. MAPPER VARIABLES ---
        self.edge = {}
        self.visited_node = []
        
        # --- 3. COMMANDER VARIABLES ---
        self.state = "EXPLORING" # EXPLORING, RESCUING, RETURNING, DROPPING
        self.path_history = {}
        self.current_target = None 
        self.rescue_center_pos = None 
        self.position_before_rescue = None
        self.initial_position = None
        self.cnt_timestep = 0
        # [THÊM MỚI] Lưu vị trí người vừa cứu thành công
        self.last_rescue_pos = None
        self.initial_spot_pos = None
        self.found_person_pos = None
        self.patience = None
        self.waypoint_stack = []
        self.not_grapsed = False
        self.drop_step = 0

        misc_data = kwargs.get('misc_data')
        self.max_timesteps = 2700 # Default fallback
        self.map_size = (800, 600) # Giá trị mặc định an toàn nếu không lấy được
        if misc_data:
            self.max_timesteps = misc_data.max_timestep_limit
            self.map_size = misc_data.size_area # Trả về (width, height)

        # # --- 4. NẠP NÃO AI (RL PILOT V12) ---
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # # Đảm bảo bạn đã đổi tên file model thành 'swarm_pilot_v12.zip'
        # model_path = os.path.join(script_dir, "swarm_pilot_v15.zip")
        
        # self.pilot_model = None
        # try:
        #     self.pilot_model = PPO.load(model_path, device='cpu') 
        #     # print(f"Drone {identifier}: Đã nạp RL Pilot V15 thành công!")
        # except Exception as e:
        #     # print(f"Drone {identifier}: LỖI nạp RL Pilot ({e}). Dùng PID dự phòng.")
        #     self.pilot_model = None

    def reset(self):
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None
        
        self.edge = {}
        self.visited_node = []
        
        self.state = "EXPLORING"
        self.path_history = {}
        self.current_target = None
        self.rescue_center_pos = None 
        self.position_before_rescue = None
        self.initial_position = None
        self.cnt_timestep = 0
        self.last_rescue_pos = None
        self.initial_spot_pos = None
        self.found_person_pos = None
        self.patience = None
        self.waypoint_stack = []

    def update_navigator(self):
        """Cập nhật vị trí ước tính từ GPS hoặc Odometer (Dead Reckoning)."""
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            self.estimated_pos = gps_pos
            self.estimated_angle = compass_angle
            self.gps_last_known = gps_pos
        else:
            # Mất GPS -> Dùng Odometer cộng dồn
            odom = self.odometer_values() # [dist, alpha, theta]
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                move_angle = self.estimated_angle + alpha
                self.estimated_pos[0] += dist * math.cos(move_angle)
                self.estimated_pos[1] += dist * math.sin(move_angle)
                self.estimated_angle = normalize_angle(self.estimated_angle + theta)
                
        if self.initial_position is None: 
            self.initial_position = self.estimated_pos

    def availability_gps(self):
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        return gps_pos is not None or compass_angle is not None

    def lidar_possible_paths(self) -> List:
        '''
        Collect Lidar data, analyze and return a list of potential areas (Frontiers).
        Modified: Sort the list to prioritize points directly IN FRONT of the drone.
        Returns an empty list if GPS is not working and there is no self.estimated_pos
        '''
        list_possible_area = []
        min_ray = -3/4 * math.pi, 0
        max_ray = 0, 0
        ray_ini = False
        minimal_distance = 170
        step_forward = 135  
        
        # Note: Should use estimated_pos instead of gps_values to avoid errors when GPS is lost
        coords = self.estimated_pos
        angle = self.estimated_angle

        if coords is None: return [] # Avoid crash if GPS is lost and estimated_pos is not set


        # Helper function to calculate angle deviation (used for sorting)
        def sort_key_by_angle(item):
            # item structure: ((x, y), visited)
            target_pos = item[0]
            dx = target_pos[0] - coords[0]
            dy = target_pos[1] - coords[1]
            
            # Angle of the vector from drone to target point
            target_vector_angle = math.atan2(dy, dx)
            # Angle deviation from drone's heading (normalized to -pi to pi)
            diff = normalize_angle(target_vector_angle - angle, False)
            
            # Return absolute value (closer to 0 is better)
            return abs(diff)

        if not self.lidar_is_disabled():
            lidar_data = self.lidar_values()
            # [FIX CRASH] Thêm dòng này vào
            if lidar_data is None: 
                return []
            ray_angles = self.lidar_rays_angles()
            
            for i in range(22, len(lidar_data) - 22):
                if lidar_data[i] > minimal_distance:
                    if lidar_data[i - 1] <= minimal_distance:
                        if i == 22:
                            ray_ini = True
                        min_ray = ray_angles[i], i
                else:
                    if i != 0 and lidar_data[i - 1] > minimal_distance:
                        max_ray = ray_angles[i - 1], i - 1
                        if max_ray != min_ray and min_ray[1] + 3 < max_ray[1]:
                            # Calculate coordinates
                            avg_angle = (min_ray[0] + max_ray[0]) / 2
                            tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                            ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                            list_possible_area.append(((tx, ty), False))
                
                # Handle edge case (circular wrap-around)
                if i == len(lidar_data) - 23 and min_ray[1] > max_ray[1]:
                    if ray_ini:
                        boolean = True
                        for k in range(min_ray[1], len(lidar_data) + 22):
                            if boolean:
                                if lidar_data[i % 181] <= minimal_distance:
                                    boolean = False

                        if boolean:
                            #del list_possible_area[0]
                            
                            # Calculate last point
                            avg_angle = (min_ray[0] + max_ray[0]) / 2
                            tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                            ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                            list_possible_area.append(((tx, ty), False))
                            
                            # --- SORT BEFORE RETURNING ---
                            list_possible_area.sort(key=sort_key_by_angle)
                            return list_possible_area

                    max_ray = ray_angles[i], i
                    # Calculate last point (no loop)
                    avg_angle = (min_ray[0] + max_ray[0]) / 2
                    tx = coords[0] + step_forward * math.cos(angle + avg_angle)
                    ty = coords[1] + step_forward * math.sin(angle + avg_angle)
                    list_possible_area.append(((tx, ty), False))

        # --- SORT BEFORE RETURNING (Normal case) ---
        list_possible_area.sort(key=sort_key_by_angle, reverse=True)
        
        return list_possible_area

    def update_mapper(self):
        """Xây dựng bản đồ các điểm đã đi qua (Graph Building)."""
        list_possible_area = self.lidar_possible_paths()
        # Dùng Int Key để tránh sai số float
        pos_key = (int(self.estimated_pos[0]), int(self.estimated_pos[1]))
        
        if pos_key not in self.edge:
            self.edge[pos_key] = [] 
            
        for val in list_possible_area:
            x = val[0][0]
            y = val[0][1]
            visited = False
            for node in self.visited_node:
                if math.hypot(x - node[0], y - node[1]) < 65.0:
                    visited = True
                    break
            if not visited: 
                self.edge[pos_key].append((x,y))
        # if len(self.edge[pos_key]) == 0: print('Không có possible path nào')
        # else: print(f'Đã thêm {len(self.edge[pos_key])} possible path mới')

    # def get_ai_observation(self, target_pos):
    #     """Chuẩn bị đầu vào 186 chiều cho Model V14 (Target Speed 15.0)."""
    #     # 1. Dist & Angle
    #     drone_pos = self.estimated_pos
    #     rel_pos = target_pos - drone_pos
    #     dist = np.linalg.norm(rel_pos)
        
    #     desired_angle = math.atan2(rel_pos[1], rel_pos[0])
    #     angle_to_target = desired_angle - self.estimated_angle
    #     angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        
    #     # 2. Velocity Info
    #     odom = self.odometer_values()
    #     if odom is not None:
    #         fwd_speed = odom[0] # px/step
    #         ang_vel = odom[1]   # rad/step
    #     else:
    #         fwd_speed = 0.0
    #         ang_vel = 0.0
            
    #     lat_speed = 0.0
        
    #     # [CẬP NHẬT CHO V15 - SPEED 12]
    #     # Hệ số chia 25.0 cho vận tốc
    #     norm_fwd = np.clip(fwd_speed / 25.0, -1.0, 1.0)
    #     norm_lat = np.clip(lat_speed / 25.0, -1.0, 1.0)
    #     norm_ang = np.clip(ang_vel, -1.0, 1.0)

    #     # 3. Lidar
    #     lidar = self.lidar_values()
    #     if lidar is None: lidar_array = np.zeros(181, dtype=np.float32)
    #     else: lidar_array = np.array(lidar, dtype=np.float32)
    #     lidar_array = np.clip(lidar_array, 0, 300.0) / 300.0
        
    #     # [QUAN TRỌNG] Vẫn giữ chia 200.0 cho khoảng cách để đảm bảo độ nhạy khi gần đích
    #     normalized_dist = np.clip(dist / 200.0, 0.0, 1.0)
        
    #     # Output: 186 inputs
    #     obs = np.concatenate(([normalized_dist, angle_to_target, norm_fwd, norm_lat, norm_ang], lidar_array)).astype(np.float32)
    #     return np.nan_to_num(obs)

    # def move_to_target(self) -> CommandsDict:
    #     if self.current_target is None:
    #         return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    #     delta_x = self.current_target[0] - self.estimated_pos[0]
    #     delta_y = self.current_target[1] - self.estimated_pos[1]
    #     dist_to_target = math.hypot(delta_x, delta_y)
        
    #     # --- LẤY VẬN TỐC ---
    #     current_speed = 0.0
    #     odom = self.odometer_values()
    #     if odom is not None:
    #         current_speed = odom[0] 
    #     else:
    #          if hasattr(self, 'prev_estimated_pos') and self.prev_estimated_pos is not None:
    #              current_speed = np.linalg.norm(self.estimated_pos - self.prev_estimated_pos)
    #     self.prev_estimated_pos = self.estimated_pos.copy()

    #     # GỌI MODEL
    #     if self.pilot_model:
    #         obs = self.get_ai_observation(self.current_target)
    #         action, _ = self.pilot_model.predict(obs, deterministic=True)
            
    #         # --- XỬ LÝ ACTION TỪ AI ---
    #         raw_forward = (action[0] + 1) / 2.0
    #         cmd_rotation = action[1]
            
    #         # --- [LAYER 1] BRAKING ASSIST (Hỗ trợ phanh từ xa) ---
    #         # Giảm dần ga khi vào vùng 80px
    #         braking_zone = 130.0
    #         if dist_to_target < braking_zone:
    #             brake_factor = np.clip(dist_to_target / braking_zone, 0.3, 1.0)
    #             cmd_forward = raw_forward * brake_factor
    #         else:
    #             cmd_forward = raw_forward

    #         # --- [LAYER 2] EMERGENCY BRAKE (Phanh gấp nếu quá nhanh ở gần) ---
    #         # Nếu còn 30px mà vẫn phi > 2.0 px/step -> Cắt ga ngay lập tức
    #         if dist_to_target <= 40.0 and current_speed > 2.0:
    #             cmd_forward = 0.1
            
    #         # --- [LAYER 3] ARRIVAL STOP (Dừng hẳn khi chạm ngưỡng) ---
    #         # Dùng luôn biến REACH_THRESHOLD (25.0) cho đồng bộ
    #         if dist_to_target <= REACH_THRESHOLD: 
    #             cmd_forward = 0.0
    #             cmd_rotation = 0.0
                
    #         return {
    #             "forward": float(cmd_forward),
    #             "lateral": 0.0,
    #             "rotation": float(cmd_rotation),
    #             "grasper": 1 if self.state in ["RESCUING", "RETURNING"] else 0
    #         }
        
    #     return self.pid_control(dist_to_target)
    
    def is_blocked_by_drone(self, safety_dist=60.0, safety_angle=0.2):
        """
        Kiểm tra xem có drone nào đang chặn ngay trước mặt không.
        - safety_dist: Khoảng cách an toàn (60px).
        - safety_angle: Góc quét phía trước (+/- 0.5 rad ~ 30 độ).
        """
        semantic_data = self.semantic_values()
        if not semantic_data: 
            return False

        for data in semantic_data:
            # Chỉ quan tâm nếu vật thể là DRONE
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                # 1. Nó có ở gần không?
                if data.distance < safety_dist:
                    # 2. Nó có ở ngay trước mặt mình không?
                    # data.angle là góc lệch so với mũi drone của mình
                    if abs(data.angle) < safety_angle:
                        return True
        return False
    
    def move_to_target_PID(self) -> CommandsDict:
        """
        Điều khiển drone đi CHÍNH XÁC đến mục tiêu.
        Chiến thuật: Đi chậm, xoay chuẩn, giảm tốc sớm.
        """
        # # # print(f'Going to {self.current_target}') # Debug nếu cần
        
        if self.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        delta_x = self.current_target[0] - self.estimated_pos[0]
        delta_y = self.current_target[1] - self.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)

        # 1. Xoay về hướng mục tiêu
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.estimated_angle)
        
        # Tăng lực xoay để chỉnh hướng nhanh hơn
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # 2. Tiến tới (LOGIC MỚI: CHẬM & CHẮC)
        
        # Cấu hình tốc độ
        MAX_SPEED = 0.6
        BRAKING_DIST = 150.0
        STOP_DIST = 15.0 # Tăng khoảng cách dừng lên một chút để an toàn

        if dist_to_target > BRAKING_DIST:
            forward_cmd = MAX_SPEED
        elif dist_to_target > STOP_DIST:
            # Giảm tốc tuyến tính
            # Bỏ dòng max(0.1, ...) để cho phép nó giảm về gần 0
            forward_cmd = (dist_to_target / BRAKING_DIST) * MAX_SPEED
            forward_cmd = max(0.1, forward_cmd)
        else:
            # Rất gần (< 15px): Cắt ga hoàn toàn
            forward_cmd = 0.05

        # 3. Kỷ luật Xoay (Strict Rotation)
        # Chỉ được phép di chuyển nếu hướng đã chuẩn (lệch < 0.2 rad ~ 11 độ)
        # Code cũ là 0.5 (30 độ) -> Quá lỏng lẻo
        if abs(angle_error) > 0.2:
            forward_cmd = 0.0 # Dừng lại để xoay cho xong đã

        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # 4. Tránh va chạm (Lidar Safety) - Giữ nguyên
        # if self.lidar_using_state:
        # lidar_vals = self.lidar_values()
        # if lidar_vals is not None:
        #     if lidar_vals[90] < SAFE_DISTANCE:
        #         forward_cmd = 0.0 
        #         rotation_cmd = 1.0 

        # --- LOGIC ĐẶC BIỆT CHO RETURNING (Cõng người) ---
        if self.grasped_wounded_persons():
            forward_cmd = 0.7
            if dist_to_target <= 60.0: forward_cmd = 0.4
        # else:
        #     # # print(f'Spec of moving, forward: {forward_cmd}, rotation: {rotation_cmd}')

        # --- [MỚI] 4. XỬ LÝ TRÁNH VA CHẠM DRONE (DEADLOCK RESOLUTION) ---
        # Nếu bị drone chặn đường -> Kích hoạt luật tay phải
        if forward_cmd > 0.05 and self.is_blocked_by_drone(safety_dist=60.0):
                # # print(f"Drone {self.identifier}: Blocked! Dodging RIGHT...")
                
                # a. Ngắt động cơ tiến để tránh va chạm
                forward_cmd = 0.0 
                
                # b. Trượt sang PHẢI (Right-hand rule)
                # lateral < 0 là sang phải, > 0 là sang trái
                cmd_lateral = -0.6 
                
                # Khi cả 2 drone cùng trượt sang phải của chính nó -> Chúng sẽ tách nhau ra

        # --- LOGIC GRASPER THÔNG MINH ---
        grasper_val = 0
        
        # Trường hợp 1: Đang mang người về (RETURNING) -> Luôn giữ chặt (1)
        if self.state == "RETURNING" or self.state == "DROPPING":
            grasper_val = 1
            
        # Trường hợp 2: Đang đi cứu (RESCUING) -> Chỉ mở càng khi đã đến gần (< 20px)
        # Để tránh bay ngang qua thằng khác rồi cướp người của nó
        elif self.state == "RESCUING":
            if dist_to_target <= 12.0:
                grasper_val = 1
            else:
                grasper_val = 0
        
        # Trường hợp 3: END_GAME (Nếu đang giữ người thì giữ chặt, đến nơi thả thì DROPPING tự lo)
        elif self.state == "END_GAME" and self.grasped_wounded_persons():
             grasper_val = 1

        if forward_cmd > 0.05 and self.is_blocked_by_drone(safety_dist=60.0):
                # # print(f"Drone {self.identifier}: Blocked by another drone! Waiting...")
                forward_cmd = 0.0 # Phanh lại chờ
                # cmd_rotation giữ nguyên để AI tiếp tục chỉnh hướng nếu cần
        if self.not_grapsed: grasper_val = 0

        # -------------------------------------------------------------
        # [NEW] ANTI-STUCK MECHANISM (Cơ chế chống kẹt tường Rescue Center)
        # -------------------------------------------------------------
        # Chỉ kích hoạt khi đang muốn vào trạm (RETURNING/END_GAME)
        # và đang ở khá gần (< 100px) nhưng chưa đến đích
        if self.state in ["RETURNING", "END_GAME"] and dist_to_target < 100.0 and dist_to_target > 30.0:
            
            # 1. Check xem có vật cản cứng ngay trước mặt không (Lidar Front)
            lidar_vals = self.lidar_values()
            if lidar_vals is not None:
                # Lấy chùm tia phía trước (góc -20 đến +20 độ)
                # Giả sử lidar có 181 tia, tia giữa là 90
                front_rays = lidar_vals[85:95] 
                min_front_dist = min(front_rays)
                
                # Nếu tường quá gần (< 35px)
                if min_front_dist < 15.0:
                    # # print(f"Drone {self.identifier}: Stuck at wall! Sliding...")
                    
                    # 2. Xử lý Trượt (Sliding)
                    # Chúng ta tắt động cơ tiến (để không húc đầu vào tường nữa)
                    forward_cmd = 0.0
                    
                    # Xác định hướng trượt dựa vào góc lệch của Target (angle_error)
                    # angle_error > 0: Target ở bên Trái -> Trượt Trái (Lateral > 0)
                    # angle_error < 0: Target ở bên Phải -> Trượt Phải (Lateral < 0)
                    
                    # Hệ số trượt:
                    # Nếu góc lệch lớn -> Trượt mạnh
                    # Nếu góc lệch nhỏ (đối diện qua tường) -> Trượt mặc định sang phải (hoặc trái) để phá thế bế tắc
                    
                    slide_force = 0.6 # Lực trượt đủ mạnh
                    
                    if abs(angle_error) < 0.1: 
                        # Trường hợp đặc biệt: Target nằm thẳng hàng sau bức tường (góc lệch ~ 0)
                        # Drone không biết nên sang trái hay phải -> Ép buộc sang Phải (Quy tắc tay phải)
                        cmd_lateral = -slide_force 
                    elif angle_error > 0:
                        cmd_lateral = slide_force  # Trượt Trái
                    else:
                        cmd_lateral = -slide_force # Trượt Phải
                        
                    # Lưu ý: rotation_cmd vẫn giữ nguyên để mũi luôn hướng về target
                    # Điều này tạo ra chuyển động xoay quanh tâm vật cản rất đẹp mắt

                    return {
                        "forward": forward_cmd, # Ngắt lực tiến
                        "lateral": cmd_lateral, # Kích hoạt lực trượt
                        "rotation": rotation_cmd, 
                        "grasper": grasper_val
                    }

        # -------------------------------------------------------------

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
                # # print(f'Add {pos_key} to visited nodes')
                self.visited_node.append(pos_key)

    def is_path_blocked(self, target_pos, safety_margin=20):
        """
        Kiểm tra xem đường thẳng từ vị trí hiện tại đến target_pos có bị chặn không.
        safety_margin: Khoảng cách an toàn (px) để không đi quá sát tường.
        """
        if target_pos is None: return False
        
        rel_pos = target_pos - self.estimated_pos
        dist = np.linalg.norm(rel_pos)
        target_angle = math.atan2(rel_pos[1], rel_pos[0])
        
        # Góc lệch so với mũi drone
        angle_diff = normalize_angle(target_angle - self.estimated_angle)
        
        # Tìm tia Lidar tương ứng với góc này
        # Lidar quét từ -135 đến +135 độ (tùy config, thường là index tương ứng)
        # ray index = (angle + 135) / step (giả sử)
        
        # Cách đơn giản hơn: Duyệt qua các tia Lidar xem tia nào trùng góc
        lidar_data = self.lidar_values()
        ray_angles = self.lidar_rays_angles()

        # [FIX CRASH] Nếu vào Kill Zone, Lidar sẽ trả về None -> Return False luôn để tránh lỗi
        if lidar_data is None or ray_angles is None:
            return False
        
        # Tìm tia gần nhất với hướng target
        min_diff = float('inf')
        closest_ray_idx = -1
        
        for i, ray_angle in enumerate(ray_angles):
            diff = abs(normalize_angle(ray_angle - angle_diff))
            if diff < min_diff:
                min_diff = diff
                closest_ray_idx = i
                
        if closest_ray_idx != -1:
            measured_dist = lidar_data[closest_ray_idx]
            # Nếu khoảng cách đo được < khoảng cách tới đích -> BỊ CHẶN
            # (Trừ đi safety_margin để không đi sát sạt tường)
            if measured_dist < (dist - safety_margin): 
                return True
                
        return False
    
    def find_best_bypass(self, original_target):
        """
        Tìm điểm trung gian (frontier) có hướng gần nhất với original_target.
        """
        possible_nodes = self.lidar_possible_paths() # Hàm này đã có sẵn
        if not possible_nodes:
            return None
            
        rel_pos = original_target - self.estimated_pos
        target_angle = math.atan2(rel_pos[1], rel_pos[0])
        
        best_node = None
        min_angle_diff = float('inf')
        
        for node_info in possible_nodes:
            # node_info cấu trúc: ((x, y), visited)
            node_pos = np.array(node_info[0])
            
            node_rel = node_pos - self.estimated_pos
            node_angle = math.atan2(node_rel[1], node_rel[0])
            
            diff = abs(normalize_angle(node_angle - target_angle))
            
            if diff < min_angle_diff:
                min_angle_diff = diff
                best_node = node_pos
                
        return best_node
    
    # ---------------------------------------------------------
    # ATC HELPER FUNCTIONS (Landing Slots & Traffic Control)
    # ---------------------------------------------------------
    def should_wait_in_queue(self):
        """
        Kiểm tra xem có ai đang được ưu tiên hơn mình không.
        Quy tắc:
        1. Chỉ so sánh với những drone đang CẠNH TRANH (RETURNING, DROPPING, END_GAME).
        2. Nếu có ai đó GẦN TÂM HƠN mình -> Mình phải chờ.
        3. Nếu khoảng cách bằng nhau -> So sánh ID (ID nhỏ đi trước) để tránh kẹt.
        """
        if self.rescue_center_pos is None: return False
        
        # Tính khoảng cách của bản thân tới trạm
        my_dist = np.linalg.norm(self.estimated_pos - self.rescue_center_pos)
        
        # Nếu mình đang ở quá xa (> 200px) thì cứ bay tự nhiên, chưa cần xếp hàng
        if my_dist > 200.0:
            return False

        if not self.communicator_is_disabled():
            for msg_package in self.communicator.received_messages:
                content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
                if not isinstance(content, dict): continue
                
                other_state = content.get("state")
                other_pos = content.get("current_pos")
                other_id = content.get("id")
                
                if other_pos is None: continue
                
                # Chỉ cạnh tranh với những kẻ đang muốn vào hoặc đang chiếm chỗ
                if other_state in ["RETURNING", "DROPPING", "END_GAME"]:
                    other_dist = np.linalg.norm(np.array(other_pos) - self.rescue_center_pos)
                    
                    # 1. Nếu nó gần hơn mình đáng kể (> 10px) -> Nhường nó
                    if other_dist < my_dist - 10.0:
                        return True
                    
                    # 2. Nếu khoảng cách xấp xỉ nhau (trong khoảng 10px) -> So ID để quyết định ai vào trước
                    if abs(other_dist - my_dist) <= 10.0:
                        if other_id < self.identifier: # Ai ID bé hơn thì được ưu tiên (người ID lớn phải chờ)
                            return True
                            
        return False # Không có ai ưu tiên hơn mình -> Được phép đi

    def control(self) -> CommandsDict:
        self.cnt_timestep += 1
        check_center = False
        
        # 1. Update Navigator
        self.update_navigator()
        if not self.availability_gps(): REACH_THRESHOLD = 30.0
        else: REACH_THRESHOLD = 25.0

        # 2. Xử lý Giao Tiếp (Communication)
        claimed_targets = []
        if not self.communicator_is_disabled():
            for msg_package in self.communicator.received_messages:
                if isinstance(msg_package, tuple):
                    _, content = msg_package
                else:
                    content = msg_package

                if isinstance(content, dict):
                    sender_state = content.get("state")
                    sender_target = content.get("target_pos")
                    if sender_state in ["RESCUING", "RETURNING"] and sender_target is not None:
                        claimed_targets.append(sender_target)
        
        # 3. Semantic Sensor (Tìm người / Trạm)
        semantic_data = self.semantic_values()
        
        # [MỚI] Biến tạm để tìm người gần nhất trong frame này
        closest_person_dist = float('inf')
        closest_person_pos = None
        
        if semantic_data:
            for data in semantic_data:
                angle_global = self.estimated_angle + data.angle
                obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                
                # --- LOGIC TÌM NGƯỜI (SỬA ĐỔI) ---
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    # So sánh: Nếu người này gần hơn người trước đó tìm thấy -> Lưu lại
                    if data.distance < closest_person_dist:
                        closest_person_dist = data.distance
                        closest_person_pos = np.array([obj_x, obj_y])
                
                # --- LOGIC TÌM TRẠM (GIỮ NGUYÊN) ---
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    check_center = True
                    if self.rescue_center_pos is None:
                       self.rescue_center_pos = np.array([obj_x, obj_y])
            # [QUAN TRỌNG] Sau khi quét hết, nếu tìm thấy người -> Gán vào biến class
            if closest_person_pos is not None:
                # Kiểm tra ngay lập tức: Người này có ai xí phần chưa?
                if not self.is_target_taken_or_better_candidate(closest_person_pos):
                    # Nếu chưa ai lấy và mình là ứng cử viên tốt nhất -> Ghi nhận
                    if self.found_person_pos is None:
                        self.found_person_pos = closest_person_pos

        steps_remaining = self.max_timesteps - self.cnt_timestep
        
        RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        
        # Đảm bảo tối thiểu 150 bước cho các map cực nhỏ (để tránh lỗi chia tỷ lệ quá bé)
        if RETURN_TRIGGER_STEPS < 450: RETURN_TRIGGER_STEPS = 500

        # FORCE RETURN: Nếu đang đi chơi mà hết giờ -> Về ngay
        if  steps_remaining <= RETURN_TRIGGER_STEPS:
            # print(f"Drone {self.identifier}: 🚨 TIME ALERT ({steps_remaining} left)! Force RETURNING.")
            self.state = "RETURNING"
            self.not_grapsed = True
            self.current_target = self.rescue_center_pos

        # 4. STATE MACHINE
        
        # --- EXPLORING ---
        if self.state == "EXPLORING":
            
            if self.found_person_pos is not None:
                # [CHECK LẦN CUỐI TRƯỚC KHI HÀNH ĐỘNG]
                # Kiểm tra lại xem trong lúc mình suy nghĩ, có thằng nào khác lao vào chưa?
                if self.is_target_taken_or_better_candidate(self.found_person_pos):
                    # print(f"Drone {self.identifier}: Hủy cứu người tại {self.found_person_pos} do có drone khác ưu tiên hơn.")
                    
                    # Đánh dấu chỗ này là đã visit để ko quay lại ngay lập tức
                    self.visit(self.found_person_pos) 
                    self.found_person_pos = None # Xóa mục tiêu để tìm người mới
                    
                    # Không return, để code chạy tiếp xuống phần tìm Frontier bên dưới
                
                else:
                    # Nếu vẫn ok -> CHỐT ĐƠN
                    self.state = "RESCUING"
                    # print(f"Drone {self.identifier}: Chuyển sang cứu người, chế độ RESCUING")
                    self.position_before_rescue = self.current_target 
                    if self.position_before_rescue is None: self.position_before_rescue = self.estimated_pos
                    
                    self.current_target = self.found_person_pos
                    self.initial_spot_pos = self.found_person_pos.copy()
            
            # LOGIC CHỌN TARGET MỚI KHI ĐÃ ĐẾN NƠI HOẶC CHƯA CÓ TARGET
            elif self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                if self.current_target is not None:
                    self.visit(self.current_target)
                else:
                    self.visit(self.estimated_pos)
                
                self.update_mapper()

                pos_key = (int(self.estimated_pos[0]), int(self.estimated_pos[1]))
                if pos_key in self.edge and len(self.edge[pos_key]):
                    next_target = self.edge[pos_key].pop()
                    target_int_key = (int(next_target[0]), int(next_target[1]))
                    
                    if self.current_target is None: 
                        self.path_history[target_int_key] = self.estimated_pos.copy()
                    else: 
                        self.path_history[target_int_key] = self.current_target.copy()
                    
                    self.current_target = np.array(next_target)
                    # print(f"Drone {self.identifier}: Chọn target mới: {self.current_target}, khoảng cách: {np.linalg.norm(self.estimated_pos - self.current_target)}, timestep: {self.cnt_timestep}")
                else:
                    # Backtracking
                    if self.current_target is not None:
                        current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                        if current_int_key in self.path_history:
                             self.current_target = self.path_history[current_int_key]
                             # print(f'Drone {self.identifier}: Đi về parent node: {self.current_target}, khoảng cách: {np.linalg.norm(self.estimated_pos - self.current_target)}, timestep: {self.cnt_timestep}')
                    else:
                        self.current_target = self.estimated_pos.copy()

            # --- [SỬA ĐOẠN NÀY] CHECK TƯỜNG CHẶN ---
            elif np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                if self.is_path_blocked(self.current_target):
                    
                    # [BƯỚC 1] Check xem có phải Drone chặn không? (Nhìn xa 100px)
                    if self.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5):
                        # Nếu là drone -> KHÔNG làm gì cả (để move_to_target_PID xử lý né)
                        pass 
                    
                    # [BƯỚC 2] Nếu không phải drone -> Chắc chắn là TƯỜNG -> Bypass
                    else:
                        # print(f"Drone {self.identifier}: ⚠️ Path to {self.current_target} BLOCKED by Wall! Finding bypass...")
                        bypass_node = self.find_best_bypass(self.current_target)
                        
                        if bypass_node is not None:
                            target_key = (int(self.current_target[0]), int(self.current_target[1]))
                            if target_key in self.path_history:
                                parent_of_blocked = self.path_history[target_key]
                                bypass_key = (int(bypass_node[0]), int(bypass_node[1]))
                                self.path_history[bypass_key] = parent_of_blocked
                                self.current_target = bypass_node
                            else:
                                self.current_target = bypass_node

        # --- RESCUING ---
        elif self.state == "RESCUING":
            # 1. Update vị trí và check "Bị di chuyển"
            if not self.grasped_wounded_persons():
                if self.patience is None: self.patience = 0
                self.patience += 1
                
                if self.patience > 30:
                    # print(f"Drone {self.identifier}: Lost target (Timeout). Back to Exploring.")
                    self.found_person_pos = None
                    self.patience = None
                    self.state = "EXPLORING"
                    self.current_target = self.position_before_rescue
                    return self.move_to_target_PID()

            # 3. Check Grasp thành công
            if self.grasped_wounded_persons():
                # print(f"Drone {self.identifier}: Grasp SUCCESS! Start RETURNING.")
                self.last_rescue_pos = self.current_target.copy()
                
                self.state = "RETURNING"
                # Lúc này position_before_rescue đóng vai trò là điểm đầu tiên của hành trình về
                self.current_target = self.rescue_center_pos

        # --- RETURNING ---
        elif self.state == "RETURNING":
            # Check xem đã về đến điểm xuất phát chưa?
            dist_to_home = np.linalg.norm(self.estimated_pos - self.rescue_center_pos)
            
            # Điều kiện chuyển sang END_GAME:
            # - Đã về rất gần nhà (< 25px)
            # - VÀ đang trong tình trạng sắp hết giờ (để phân biệt với việc về nhà cất người xong đi tiếp)
            if dist_to_home < 50 and steps_remaining <= RETURN_TRIGGER_STEPS:
                # print(f"Drone {self.identifier}: Đã về nhà an toàn. Chuyển sang END_GAME.")
                self.state = "END_GAME"
                self.current_target = None # Reset target để END_GAME tự xử lý
                return self.move_to_target_PID()
            
            if check_center and self.rescue_center_pos is not None and steps_remaining > RETURN_TRIGGER_STEPS:
                if not self.grasped_wounded_persons(): 
                    # print(f"Drone {self.identifier}: Chuyển sang DROPPING do va vào rescue center trên đường")
                    self.state = "DROPPING"
                # Target bây giờ là CHÍNH TÂM TRẠM
                self.current_target = self.rescue_center_pos
                
                # Kiểm tra xem có cần xếp hàng không?
                if self.should_wait_in_queue():
                    # # print(f"Drone {self.identifier}: Đang xếp hàng... (Nhường drone khác)")
                    
                    # Dừng lại chờ (hoặc lùi nhẹ nếu đứng quá sát < 80px để nhường chỗ cho con bên trong đi ra)
                    dist_to_center = np.linalg.norm(self.estimated_pos - self.rescue_center_pos)
                    
                    forward_val = -1
                    
                    return {
                        "forward": forward_val, 
                        "lateral": 0.0, 
                        "rotation": 0.0,
                        "grasper": 1
                    }

                # Nếu KHÔNG phải chờ -> Lao thẳng vào tâm
                # Nếu đến rất gần (< 15px) -> DROPPING
                if np.linalg.norm(self.estimated_pos - self.current_target) < 20.0:
                    self.state = "DROPPING"
                    self.drop_step = 0
            else:
                if self.current_target is None: self.current_target = self.position_before_rescue
                # --- [LOGIC MỚI] CHECK SHORTCUT ---
                # Chỉ check khi đang di chuyển (khoảng cách > REACH_THRESHOLD)
                # và cứ mỗi 5 timestep check 1 lần cho đỡ lag
                if self.cnt_timestep % 5 == 0 and np.linalg.norm(self.estimated_pos - self.current_target) > REACH_THRESHOLD:
                    shortcut = self.find_shortcut_target()
                    if shortcut is not None:
                        # Nếu tìm thấy đường tắt, cập nhật target luôn
                        # Lưu ý: Khi nhảy cóc, ta vẫn phải cập nhật Stack để Commuting hoạt động đúng
                        # Nhưng vì nhảy cóc nên Stack sẽ thưa hơn -> Drone quay lại cũng nhanh hơn
                        
                        # (Optional) Nếu muốn Stack chính xác từng bước thì phải push cả đoạn giữa vào
                        # Nhưng ở đây ta chấp nhận Stack thưa để tối ưu cả chiều đi và về
                        self.current_target = shortcut
                if np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                    # Logic Backtracking cũ
                    if self.current_target is not None:
                        current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                        
                        if current_int_key in self.path_history: 
                            parent_node = self.path_history[current_int_key]
                            # [THÊM MỚI] Ghi lại node hiện tại vào stack trước khi chuyển sang cha
                            self.waypoint_stack.append(self.current_target.copy())
                            self.current_target = parent_node 
                        else:
                            if self.rescue_center_pos is not None:
                                self.current_target = self.rescue_center_pos
                        # print(f'Drone {self.identifier}: Trở về nhà: {self.current_target}, khoảng cách: {np.linalg.norm(self.estimated_pos - self.current_target)}, timestep: {self.cnt_timestep}')
                
                # --- [SỬA ĐOẠN NÀY] CHECK TƯỜNG CHẶN ---
                elif np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                    if self.is_path_blocked(self.current_target):
                        
                        # [MỚI] Phân loại vật cản
                        if self.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5):
                            pass # Là drone -> Chờ nó đi hoặc né nhau
                        else:
                            # Là tường -> Tìm đường vòng
                            # print(f"Drone {self.identifier}: ⚠️ Returning Path BLOCKED by Wall! Finding bypass...")
                            bypass_node = self.find_best_bypass(self.current_target)
                            if bypass_node is not None:
                                target_key = (int(self.current_target[0]), int(self.current_target[1]))
                                if target_key in self.path_history:
                                    parent_of_blocked = self.path_history[target_key]
                                    bypass_key = (int(bypass_node[0]), int(bypass_node[1]))
                                    self.path_history[bypass_key] = parent_of_blocked
                                    self.current_target = bypass_node
                                else:
                                    self.current_target = bypass_node

        # --- DROPPING ---
        elif self.state == "DROPPING":
            self.drop_step += 1
            if self.drop_step > 100:
                self.state = "INITIAL"
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            # 1. Kiểm tra xem nạn nhân đã thực sự rơi ra chưa?
            if self.grasped_wounded_persons():
                # rint(f"Drone {self.identifier}: Stuck payload! Spinning to release...")
                return {
                    "forward": -0.3,  # [SỬA] Lùi nhẹ để tách khỏi tường/trạm
                    "lateral": 0.0,   
                    "rotation": 0.0,  # Xoay max
                    "grasper": 0      
                }
            else:
                # print(f"Drone {self.identifier}: Drop SUCCESS! Soft Reset & Back to Base.")
                self.state = "INITIAL"
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- INITIAL ---
        elif self.state == "INITIAL":
            # Chỉ reset các biến tạm thời của việc cứu hộ
            self.last_rescue_pos = None 
            self.initial_spot_pos = None 
            self.found_person_pos = None
            # self.position_before_rescue = None # Giữ lại cái này nếu cần debug, ko quan trọng
            
            # print(f"Drone {self.identifier}: Bắt đầu quay lại chỗ cũ (COMMUTING)...")
            
            # Chuyển sang trạng thái đi làm lại
            self.state = "COMMUTING"
            self.current_target = None # Để logic COMMUTING tự lấy target đầu tiên
            
            # Lùi nhẹ để tách khỏi đám đông ở trạm
            return {"forward": -0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- COMMUTING (Đi quay lại chỗ cũ theo đường đã về) ---
        elif self.state == "COMMUTING":
            # 1. Nếu chưa có target hoặc đã đến target hiện tại
            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                
                # Nếu còn điểm trong stack -> Lấy ra đi tiếp
                if len(self.waypoint_stack) > 0:
                    next_waypoint = self.waypoint_stack.pop() # Lấy điểm mới nhất ra (LIFO)
                    self.current_target = next_waypoint
                    # print(f"Drone {self.identifier}: Commuting to: {self.current_target} (Stack left: {len(self.waypoint_stack)})")
                
                # Nếu hết stack -> Đã đến nơi (position_before_rescue)
                else:
                    # print(f"Drone {self.identifier}: Đã quay lại điểm xuất phát! Chuyển sang EXPLORING.")
                    self.state = "EXPLORING"
                    # Gán target về chính chỗ đó để EXPLORING tiếp tục mở rộng từ đây
                    self.current_target = self.position_before_rescue 
            
            # 2. Logic check tường & Bypass
            elif np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                if self.is_path_blocked(self.current_target):
                    
                    # [MỚI] Check Drone
                    if self.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5):
                        pass 
                    else:
                        # print(f"Drone {self.identifier}: ⚠️ Commuting Path Blocked! Finding bypass...")
                        bypass_node = self.find_best_bypass(self.current_target)
                        if bypass_node is not None:
                             # Bypass cho Commuting ko cần update history ngược
                             self.current_target = bypass_node
            
        # --- END_GAME ---
        elif self.state == "END_GAME":
            # if self.grasped_wounded_persons():
            #     # Target là tâm trạm
            #     target_center = self.rescue_center_pos if self.rescue_center_pos is not None else self.initial_position
            #     self.current_target = target_center
                
            #     # Logic Xếp hàng cho End Game
            #     if self.rescue_center_pos is not None and self.should_wait_in_queue():
            #         # # print(f"Drone {self.identifier}: EndGame Queue Waiting...")
                    
            #         dist_to_center = np.linalg.norm(self.estimated_pos - self.rescue_center_pos)
            #         forward_val = 0.0
            #         if dist_to_center < 90.0: forward_val = -0.1
                        
            #         return {
            #             "forward": forward_val, "lateral": 0.0, "rotation": 0.0, "grasper": 1
            #         }
            
            # else:
            #     # Logic về chỗ nằm chờ (Giữ nguyên)
            #     self.current_target = self.initial_position
            #     if np.linalg.norm(self.estimated_pos - self.initial_position) < 10.0:
            #         return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return self.move_to_target_PID()

        return self.move_to_target_PID()

    def define_message_for_all(self):
        # Xác định vị trí người mà drone này đang quan tâm
        # Nếu đang RESCUING/RETURNING -> Là current_target
        # Nếu đang EXPLORING mà vừa tìm thấy -> Là found_person_pos
        person_target = None
        
        if self.state in ["RESCUING", "RETURNING"]:
            # Khi đang cứu hoặc về, target chính là người (hoặc vị trí người đã cứu)
            # Lưu ý: RETURNING thì current_target là Rescue Center, nên phải dùng last_rescue_pos
            if self.state == "RETURNING":
                person_target = self.last_rescue_pos
            else:
                person_target = self.current_target
        elif self.state == "EXPLORING":
            person_target = self.found_person_pos

        msg_data = {
            "id": self.identifier,
            "state": self.state,
            "person_pos": person_target, # [QUAN TRỌNG] Tọa độ người đang được nhắm tới
            "current_pos": self.estimated_pos
        }
        return msg_data
    
    def is_target_taken_or_better_candidate(self, target_person_pos):
        """
        Kiểm tra xem người tại vị trí target_person_pos:
        1. Đã có ai đang cứu (RESCUING) hoặc đang mang về (RETURNING) chưa?
        2. Có ai cũng đang nhìn thấy (EXPLORING) nhưng đứng GẦN HƠN mình không?
        """
        if target_person_pos is None: return False
        if self.communicator_is_disabled(): return False
        
        # Khoảng cách từ mình đến người đó
        my_dist = np.linalg.norm(self.estimated_pos - target_person_pos)
        
        # Ngưỡng sai số tọa độ (vì sensor có nhiễu, 2 drone nhìn 1 người có thể lệch nhau vài chục px)
        COORDINATE_MATCH_THRESHOLD = 50.0 

        for msg_package in self.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            other_id = content.get("id")
            other_state = content.get("state")
            other_person_pos = content.get("person_pos") # Vị trí người mà drone kia đang nhắm tới
            other_current_pos = content.get("current_pos")
            
            if other_person_pos is None: continue
            
            # Kiểm tra xem drone kia có đang nhắm tới CÙNG MỘT NGƯỜI với mình không
            dist_between_targets = np.linalg.norm(target_person_pos - other_person_pos)
            
            if dist_between_targets < COORDINATE_MATCH_THRESHOLD:
                # --- TRƯỜNG HỢP 1: Người đó đã bị thằng khác CHỐT ĐƠN ---
                if other_state in ["RESCUING", "RETURNING", "DROPPING"]:
                    return True # Bỏ đi, tìm người khác
                
                # --- TRƯỜNG HỢP 2: Cạnh tranh công bằng (Cả 2 cùng vừa nhìn thấy) ---
                if other_state == "EXPLORING" and other_current_pos is not None:
                    other_dist_to_person = np.linalg.norm(other_current_pos - other_person_pos)
                    
                    # Nếu nó gần người đó hơn mình -> Nhường nó
                    if other_dist_to_person < my_dist - 10.0: # (Trừ 10px để tránh nhiễu)
                        return True
                    
                    # Nếu khoảng cách ngang nhau -> So ID để tránh deadlock
                    if abs(other_dist_to_person - my_dist) <= 10.0:
                        if other_id < self.identifier: # ID nhỏ hơn được ưu tiên
                            return True
                            
        return False # Không ai tranh -> Mình cứu!
    
    def find_shortcut_target(self):
        """
        Tìm tổ tiên xa nhất mà drone có thể bay thẳng tới (không bị tường chặn).
        Giúp drone về nhà nhanh hơn thay vì đi lần lượt từng bước.
        """
        if self.current_target is None: return None
        
        # 1. Truy xuất chuỗi tổ tiên (Ancestors Chain)
        # Chúng ta sẽ nhìn xa tối đa khoảng 5-10 bước để đỡ tốn chi phí tính toán
        ancestors = []
        curr_key = (int(self.current_target[0]), int(self.current_target[1]))
        
        # Lấy tối đa 8 đời tổ tiên
        temp_key = curr_key
        for _ in range(8):
            if temp_key in self.path_history:
                parent_pos = self.path_history[temp_key]
                ancestors.append(parent_pos)
                temp_key = (int(parent_pos[0]), int(parent_pos[1]))
            else:
                break
        
        if not ancestors: return None

        # 2. Duyệt từ xa về gần (Greedy)
        # Check người xa nhất trước. Nếu đi được thì chốt luôn.
        for target_pos in reversed(ancestors):
            # Kiểm tra khoảng cách: Nếu quá xa (> 300px) thì thôi, vì Lidar ko quét tới đó để check tường được
            dist = np.linalg.norm(target_pos - self.estimated_pos)
            if dist > 300.0: continue 

            # Kiểm tra tường chắn
            # Lưu ý: Cần safety_margin lớn chút (30px) để đảm bảo đường tắt thực sự an toàn
            if not self.is_path_blocked(target_pos, safety_margin=30):
                # # print(f"Drone {self.identifier}: Found SHORTCUT to {target_pos}!")
                return target_pos
                
        return None # Không tìm được đường tắt nào ngon hơn current_target  