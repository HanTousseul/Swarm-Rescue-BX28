import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

# --- STABLE CONFIGURATION (STABLE CONFIGURATION) ---
SAFE_DISTANCE = 35.0 # Increase safe distance (previously 25)
KP_ROTATION = 2.0 # Moderate rotation, good control
KP_FORWARD = 0.6 # Average speed (Reduced from 1.0 to 0.6 to avoid collision)
MAX_LIDAR_RANGE = 200 # Road scanning range
EXPLORE_STEP = 140 # Shorter step (80px) to ensure the road is always in Lidar range
REACH_THRESHOLD = 25.0
TARGET_TIMEOUT = 120 # Timeout earlier to avoid locking time

class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # Navigator
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.initialized = False
        
        # Mapper
        self.frontier_stack: List[Tuple[np.ndarray, np.ndarray]] = [] 
        self.visited_grid = set() 
        self.bad_targets = [] 
        
        # Commander
        self.state = "EXPLORING" 
        self.current_target = None 
        self.branch_point = None 
        
        self.path_history = {} 
        self.last_grid_pos = None
        self.rescue_center_pos = None 
        
        # Unstuck & Debug
        self.target_start_step = 0
        self.step_counter = 0
        self.stuck_counter = 0
        self.last_pos_check = np.array([0.0, 0.0])
        self.is_recovering = False
        self.recover_step = 0

    # =========================================================================
    # MODULE 1: NAVIGATOR
    # =========================================================================
    def update_navigator(self):
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        source = "ODOM"
        if gps_pos is not None and compass_angle is not None:
            self.estimated_pos = gps_pos
            self.estimated_angle = compass_angle
            source = "GPS"
        else:
            odom = self.odometer_values() 
            if odom is not None:
                dist, alpha, theta = odom[0], odom[1], odom[2]
                move_angle = self.estimated_angle + alpha
                self.estimated_pos[0] += dist * math.cos(move_angle)
                self.estimated_pos[1] += dist * math.sin(move_angle)
                self.estimated_angle = normalize_angle(self.estimated_angle + theta)
        
        # Khởi tạo check kẹt
        if not self.initialized and (gps_pos is not None or self.estimated_pos is not None):
            self.last_pos_check = self.estimated_pos.copy()
            self.last_grid_pos = self.estimated_pos.copy()
            self.initialized = True

        if self.step_counter % 50 == 0:
            print(f"[NAV] Step {self.step_counter} | Pos: ({self.estimated_pos[0]:.0f}, {self.estimated_pos[1]:.0f}) | Src: {source}")

    # =========================================================================
    # MODULE 2: MAPPER
    # =========================================================================
    def update_mapper(self):
        lidar_vals = self.lidar_values()
        ray_angles = self.lidar().ray_angles 
        if lidar_vals is None: return

        if self.size_area:
            map_w, map_h = self.size_area
            limit_x_max, limit_x_min = (map_w/2)-40, -(map_w/2)+40
            limit_y_max, limit_y_min = (map_h/2)-40, -(map_h/2)+40
        else:
            limit_x_max, limit_x_min, limit_y_max, limit_y_min = 500, -500, 500, -500

        grid_pos = (int(self.estimated_pos[0] // 40), int(self.estimated_pos[1] // 40))
        if grid_pos not in self.visited_grid:
            self.visited_grid.add(grid_pos)

        if self.branch_point is not None:
            return

        current_node_pos = self.current_target if self.current_target is not None else self.estimated_pos.copy()

        step = 10
        for i in range(0, len(lidar_vals), step):
            dist = lidar_vals[i]
            
            if dist > MAX_LIDAR_RANGE:
                # Safety margin rộng hơn (5 tia ~ 10 độ)
                if lidar_vals[max(0, i-5)] < SAFE_DISTANCE + 30 or lidar_vals[min(180, i+5)] < SAFE_DISTANCE + 30:
                    continue

                global_angle = self.estimated_angle + ray_angles[i]
                target_x = self.estimated_pos[0] + EXPLORE_STEP * math.cos(global_angle)
                target_y = self.estimated_pos[1] + EXPLORE_STEP * math.sin(global_angle)
                
                if not (limit_x_min <= target_x <= limit_x_max and limit_y_min <= target_y <= limit_y_max):
                    continue 

                if self.rescue_center_pos is not None:
                    if np.linalg.norm(np.array([target_x, target_y]) - self.rescue_center_pos) < 50:
                        continue

                frontier_grid = (int(target_x // 40), int(target_y // 40))
                if frontier_grid not in self.visited_grid:
                    is_duplicate = False
                    
                    # --- KIỂM TRA BLACKLIST (Bán kính rộng 80px) ---
                    for bad in self.bad_targets:
                        if np.linalg.norm(np.array([target_x, target_y]) - bad) < 80:
                            is_duplicate = True; break
                    if is_duplicate: continue
                    # -----------------------------------------------

                    if self.current_target is not None:
                         if np.linalg.norm(self.current_target - np.array([target_x, target_y])) < 50:
                            is_duplicate = True
                    
                    if not is_duplicate:
                        for f_tuple in reversed(self.frontier_stack[-5:]): 
                            if np.linalg.norm(f_tuple[0] - np.array([target_x, target_y])) < 50: 
                                is_duplicate = True; break

                    if not is_duplicate:
                        self.frontier_stack.append((np.array([target_x, target_y]), current_node_pos.copy()))

    # =========================================================================
    # MODULE 3: DRIVER
    # =========================================================================
    def move_to_target(self) -> CommandsDict:
        if self.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        delta_x = self.current_target[0] - self.estimated_pos[0]
        delta_y = self.current_target[1] - self.estimated_pos[1]
        dist_to_target = math.hypot(delta_x, delta_y)
        target_angle = math.atan2(delta_y, delta_x)
        angle_error = normalize_angle(target_angle - self.estimated_angle)
        
        rotation_cmd = KP_ROTATION * angle_error
        rotation_cmd = max(-1.0, min(1.0, rotation_cmd))

        # --- TỐC ĐỘ AN TOÀN ---
        if dist_to_target > 50:
            if abs(angle_error) < 0.2: 
                forward_cmd = KP_FORWARD # Đi thẳng
            elif abs(angle_error) < 0.8: 
                forward_cmd = KP_FORWARD * 0.4 # Vừa đi vừa xoay chậm
            else: 
                forward_cmd = 0.0 # Dừng hẳn để xoay (An toàn nhất)
        else:
            forward_cmd = 0.15 # Rất chậm khi đến đích
        
        forward_cmd = max(-1.0, min(1.0, forward_cmd))

        # Unstuck
        if self.step_counter % 20 == 0:
            if np.linalg.norm(self.estimated_pos - self.last_pos_check) < 5.0 and not self.is_recovering:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_pos_check = self.estimated_pos.copy()

        if self.stuck_counter > 3:
            self.is_recovering = True
            self.stuck_counter = 0
            self.recover_step = 30

        if self.is_recovering:
            self.recover_step -= 1
            if self.recover_step <= 0: self.is_recovering = False
            # Lùi và xoay ngẫu nhiên
            return {"forward": -0.7, "lateral": 0.0, "rotation": 1.0 if self.step_counter % 10 < 5 else -1.0, "grasper": 1 if self.state in ["RESCUING", "RETURNING"] else 0}

        # Safety
        lidar_vals = self.lidar_values()
        if lidar_vals is not None:
            min_front = np.min(lidar_vals[70:110])
            
            # GIẢM TỐC CHỦ ĐỘNG
            if min_front < 80: forward_cmd = min(forward_cmd, 0.3)

            if min_front < SAFE_DISTANCE:
                forward_cmd = 0.0 # Dừng trước tường
                
                space_left = np.sum(lidar_vals[0:90])
                space_right = np.sum(lidar_vals[90:180])
                
                if space_left > space_right:
                    rotation_cmd = 0.8 
                else:
                    rotation_cmd = -0.8
                
                if min_front < 15: 
                    forward_cmd = -0.5
                    rotation_cmd = 0.0

        grasper_val = 1 if (self.state == "RESCUING" or self.state == "RETURNING") else 0
        return {"forward": forward_cmd, "lateral": 0.0, "rotation": rotation_cmd, "grasper": grasper_val}

    # --- HÀM KIỂM TRA ĐƯỜNG NGẮM (LINE OF SIGHT) ---
    def check_path_blocked(self):
        if self.current_target is None: return False
        dx = self.current_target[0] - self.estimated_pos[0]
        dy = self.current_target[1] - self.estimated_pos[1]
        dist_to_target = math.hypot(dx, dy)
        target_angle_global = math.atan2(dy, dx)
        angle_relative = normalize_angle(target_angle_global - self.estimated_angle)
        
        lidar_vals = self.lidar_values()
        ray_angles = self.lidar().ray_angles
        if lidar_vals is None: return False

        # Tìm tia Lidar gần hướng mục tiêu nhất
        closest_idx = (np.abs(ray_angles - angle_relative)).argmin()
        obstacle_dist = lidar_vals[closest_idx]
        
        # Nếu vật cản gần hơn đích (trừ biên an toàn) -> Bị chặn
        if obstacle_dist < dist_to_target - 20.0: 
            return True
        return False

    def perform_backtracking(self):
        curr_grid = (int(self.estimated_pos[0] // 40), int(self.estimated_pos[1] // 40))
        best_parent = None
        
        if curr_grid in self.path_history:
            best_parent = self.path_history[curr_grid]
        else:
            min_dist = 10000
            best_key = None
            for grid_key, parent_pos in self.path_history.items():
                real_pos = np.array([grid_key[0]*40 + 20, grid_key[1]*40 + 20])
                d = np.linalg.norm(self.estimated_pos - real_pos)
                if d < min_dist:
                    min_dist = d
                    best_key = grid_key
                    best_parent = parent_pos
            
            if best_key is not None and min_dist < 200:
                pass 
        return best_parent

    # =========================================================================
    # MODULE 4: COMMANDER
    # =========================================================================
    def control(self) -> CommandsDict:
        self.step_counter += 1
        self.update_navigator()
        self.update_mapper()
        
        semantic_data = self.semantic_values()
        found_person_pos = None
        found_rescue_pos = None
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    angle_global = self.estimated_angle + data.angle
                    px = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    py = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    found_person_pos = np.array([px, py])
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    if self.rescue_center_pos is None:
                        angle_global = self.estimated_angle + data.angle
                        rx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                        ry = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                        self.rescue_center_pos = np.array([rx, ry])
                        print(f"[SEM] Found Rescue Center: ({rx:.0f}, {ry:.0f})")
                    found_rescue_pos = True

        # Path History
        curr_grid = (int(self.estimated_pos[0] // 40), int(self.estimated_pos[1] // 40))
        if self.last_grid_pos is not None:
            if curr_grid not in self.path_history:
                last_grid = (int(self.last_grid_pos[0] // 40), int(self.last_grid_pos[1] // 40))
                if curr_grid != last_grid:
                    self.path_history[curr_grid] = self.last_grid_pos.copy()
        self.last_grid_pos = self.estimated_pos.copy()

        # --- STATE MACHINE ---
        if self.state == "EXPLORING":
            if found_person_pos is not None:
                print(f"[CMD] FOUND PERSON -> RESCUING")
                self.state = "RESCUING"
                self.current_target = found_person_pos
                self.branch_point = None
            else:
                # 1. Kiểm tra đường bị chặn (NEW FEATURE)
                if self.current_target is not None:
                    if self.check_path_blocked():
                        print(f"[CMD] PATH BLOCKED to {self.current_target}. Blacklisting!")
                        self.bad_targets.append(self.current_target.copy())
                        self.current_target = None # Buộc chọn lại

                # 2. Logic Timeout
                is_timeout = self.current_target is not None and (self.step_counter - self.target_start_step > TARGET_TIMEOUT)
                if is_timeout:
                    print(f"[CMD] TIMEOUT. Blacklisting target: {self.current_target}")
                    self.bad_targets.append(self.current_target.copy())
                    self.current_target = None
                    # Lùi lại một chút để thoát vùng ám ảnh
                    self.is_recovering = True; self.recover_step = 10

                is_reached = self.current_target is not None and np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD

                if self.current_target is None or is_reached:
                    if self.branch_point is None:
                        found_valid = False
                        while self.frontier_stack:
                            next_target, origin_point = self.frontier_stack.pop()
                            
                            # Check Blacklist (80px)
                            is_bad = False
                            for bad in self.bad_targets:
                                if np.linalg.norm(next_target - bad) < 80: is_bad = True; break
                            if is_bad: continue

                            found_valid = True
                            dist_to_origin = np.linalg.norm(self.estimated_pos - origin_point)
                            
                            if dist_to_origin < 100: 
                                self.current_target = next_target
                                self.target_start_step = self.step_counter
                                print(f"[CMD] Exploring: ({next_target[0]:.0f}, {next_target[1]:.0f})")
                            else:
                                self.branch_point = origin_point
                                self.current_target = None 
                                self.frontier_stack.append((next_target, origin_point)) 
                                print(f"[CMD] Relocating to Branch: ({origin_point[0]:.0f}, {origin_point[1]:.0f})")
                            break
                        
                        if not found_valid and not self.frontier_stack:
                            print("[CMD] Map Explored -> RETURNING")
                            self.state = "RETURNING"
                    else:
                        dist_to_branch = np.linalg.norm(self.estimated_pos - self.branch_point)
                        if dist_to_branch < REACH_THRESHOLD + 30:
                            print(f"[CMD] Arrived at Branch Point!")
                            self.branch_point = None
                            self.current_target = None
                        else:
                            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                                next_parent = self.perform_backtracking()
                                if next_parent is not None:
                                    self.current_target = next_parent
                                else:
                                    self.current_target = self.branch_point

        elif self.state == "RESCUING":
            if found_person_pos is not None: self.current_target = found_person_pos
            if self.grasped_wounded_persons():
                print(f"[CMD] GRASPED! -> RETURNING")
                self.state = "RETURNING"
                self.current_target = None

        elif self.state == "RETURNING":
            if found_rescue_pos:
                dist_to_rescue = np.linalg.norm(self.estimated_pos - self.rescue_center_pos)
                if dist_to_rescue < REACH_THRESHOLD + 40:
                    self.state = "DROPPING"
                    return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}

            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD:
                next_parent = self.perform_backtracking()
                if next_parent is not None:
                    self.current_target = next_parent
                else:
                    target = self.rescue_center_pos if self.rescue_center_pos is not None else np.array([0.0, 0.0])
                    self.current_target = target

        elif self.state == "DROPPING":
            if not self.grasped_wounded_persons():
                print(f"[CMD] DROPPED! -> EXPLORING")
                self.state = "EXPLORING" 
                self.current_target = None
                self.branch_point = None 
                self.bad_targets = [] 
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        return self.move_to_target()

    def define_message_for_all(self):
        pass