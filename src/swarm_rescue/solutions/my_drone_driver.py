import math
import random 
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

# IMPORT COMPONENTS
try:
    from .navigator import Navigator
    from .pilot import Pilot
    from .communicator import CommunicatorHandler
except ImportError:
    from navigator import Navigator
    from pilot import Pilot
    from communicator import CommunicatorHandler

class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # Config
        misc_data = kwargs.get('misc_data')
        self.max_timesteps = 2700 
        self.map_size = (1113, 750) 
        if misc_data:
            self.max_timesteps = misc_data.max_timestep_limit
            self.map_size = misc_data.size_area

        # --- INITIALIZE COMPONENTS ---
        self.nav = Navigator(self)
        self.pilot = Pilot(self)
        self.comms = CommunicatorHandler(self)
        
        # --- STATE VARIABLES ---
        self.state = "DISPERSING" # Bắt đầu bằng chế độ "Nổ"
        self.last_state = ""
        
        self.current_target = None 
        self.rescue_center_pos = None 
        self.position_before_rescue = None
        self.initial_position = None # Luôn là vị trí gốc ban đầu
        self.cnt_timestep = 0
        
        self.last_rescue_pos = None
        self.not_grapsed = False
        self.drop_step = 0

        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None 
        
        # Anti-stuck history
        self.pos_history_long = []
        self.outgoing_msg_buffer = None
        
        self.wait_scan_map = 0
        self.patience = 0
        self.blacklisted_targets = []
        self.blacklist_timer = 0

        # SWARM VARIABLES
        self.busy_targets = [] 
        # Khởi động lệch pha: 30 tick nổ + 70 tick xoay = 100 tick chuẩn bị
        self.start_moving_time = 100 + (self.identifier * 20) 
        self.rescue_time = 0

    def control(self) -> CommandsDict:
        self.cnt_timestep += 1
        
        if self.state != self.last_state:
            print(f"[{self.identifier}] 🔄 STATE CHANGE: {self.last_state} -> {self.state}")
            self.last_state = self.state

        # 1. Thu thập vị trí đồng đội
        nearby_drones_pos = []
        semantic_data = self.semantic_values()
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    angle_global = self.estimated_angle + data.angle
                    dx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    dy = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    nearby_drones_pos.append(np.array([dx, dy]))

        # 2. Update Navigator
        self.nav.update_navigator(nearby_drones=nearby_drones_pos)
        
        # Lưu vị trí gốc ban đầu (Chỉ làm 1 lần ở tick 1)
        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()
            print(f"[{self.identifier}] 🏁 STARTED at {self.initial_position}")

        # 3. Check Rescue Center
        if semantic_data:
            tmp = float('inf')
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    angle_global = self.estimated_angle + data.angle
                    obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    dist_to_center = np.linalg.norm(self.estimated_pos - np.array([obj_x, obj_y]))
                                                    
                    if dist_to_center < tmp:
                        tmp = dist_to_center
                        self.rescue_center_pos = np.array([obj_x, obj_y])

        # 4. Handle Communication
        self.comms.process_incoming_messages()
        self.comms.broadcast_map_updates() 
        self.comms.broadcast_obstacle_update() 

        # Clean up busy targets
        self.busy_targets = [t for t in self.busy_targets if t['timer'] > 0]
        for t in self.busy_targets: t['timer'] -= 1

        # Visualization (Drone 0 only)
        if self.cnt_timestep % 5 == 0:
            self.nav.obstacle_map.display(
                self.estimated_pos, 
                current_target=self.current_target,
                current_path=self.nav.current_astar_path, 
                window_name=f"Obstacle Map - Drone {self.identifier}"
            )

        # =========================================================
        # PHASE 0: DISPERSING (VỤ NỔ BIG BANG) - 30 Ticks đầu
        # =========================================================
        if self.state == "DISPERSING":
            if self.cnt_timestep < 40:
                # A. Lực đẩy từ các Drone khác (Lateral Repulsion)
                _, lat_drone = self.pilot.calculate_repulsive_force()
                
                # B. Lực đẩy từ Rescue Center (Radial Repulsion)
                # Tính vector từ Tâm (Initial Pos) hướng ra Drone
                center = self.initial_position
                dx = self.estimated_pos[0] - center[0]
                dy = self.estimated_pos[1] - center[1]
                
                # Góc hướng ra ngoài
                angle_away = math.atan2(dy, dx) 
                
                # Chuyển sang hệ tọa độ của Drone (Body Frame)
                # Để biết cần lái Forward/Lateral bao nhiêu để bay theo hướng đó
                angle_diff = angle_away - self.estimated_angle
                # Normalize về [-pi, pi]
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                
                # Phân tích lực đẩy tâm thành Forward/Lateral
                push_fwd = math.cos(angle_diff)
                push_lat = math.sin(angle_diff)
                
                # C. Tổng hợp lực (Force Blending)
                # Ưu tiên bay ra xa tâm (Push Fwd/Lat) + Né đồng đội (Lat Drone)
                final_fwd = push_fwd * 1.0 
                final_lat = (push_lat * 1.0) + (lat_drone * 5.0) # Né đồng đội cực mạnh (x5)
                
                # Thêm chút ngẫu nhiên để phá vỡ đội hình
                jitter = random.uniform(-0.5, 0.5)
                
                return {
                    "forward": np.clip(final_fwd, -1, 1), 
                    "lateral": np.clip(final_lat + jitter, -1, 1), 
                    "rotation": 0.0, # Không xoay, tập trung bay tản ra
                    "grasper": 0
                }
            else:
                # Hết giờ nổ -> Chuyển sang xoay map
                # [QUAN TRỌNG] KHÔNG CẬP NHẬT initial_position!
                # Vẫn giữ initial_position là điểm xuất phát ban đầu để sau này về đúng chỗ.
                print(f"[{self.identifier}] 💥 DISPERSION DONE. Scanning...")
                self.state = "STARTUP" 

        # =========================================================
        # PHASE 1: STARTUP (XOAY & CHỜ LỆCH PHA)
        # =========================================================
        if self.state == "STARTUP":
            # Xoay tại chỗ 70 tick để quét map (tính từ tick 30 -> 100)
            if self.cnt_timestep <= 100:
                return {"forward": 0.0, "lateral": 0.0, "rotation": 1.0, "grasper": 0}
            
            # Chờ đến giờ hoàng đạo của riêng mình
            elif self.cnt_timestep < self.start_moving_time:
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            
            else:
                print(f"[{self.identifier}] 🚀 LAUNCHING!")
                self.state = "EXPLORING"

        # 5. Check Health / Return
        steps_remaining = self.max_timesteps - self.cnt_timestep
        RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        
        if steps_remaining <= RETURN_TRIGGER_STEPS:
            if self.is_inside_return_area: self.state = "END_GAME"
            else:
                if self.state not in ["RETURNING", "DROPPING", "END_GAME"]:
                    print(f"[{self.identifier}] 🔋 LOW BATTERY! Returning home.")
                    self.state = "RETURNING"
                    self.current_target = None 
                    self.not_grapsed = True

        # Anti-stuck logic
        self.pos_history_long.append(self.estimated_pos.copy())
        waiting = 130 if self.state == 'EXPLORING' else 150
        if len(self.pos_history_long) > waiting: self.pos_history_long.pop(0) 
        
        if self.state not in ["END_GAME", "DISPERSING", "STARTUP"] and len(self.pos_history_long) == waiting and steps_remaining > RETURN_TRIGGER_STEPS:
            start_pos = self.pos_history_long[0]
            dist_moved = np.linalg.norm(self.estimated_pos - start_pos)
            if dist_moved < 8.0:
                print(f"[{self.identifier}] ⚠️ STUCK DETECTED! Replanning...")
                self.nav.current_astar_path = []
                fwd = 0
                lat = 1.0 if random.random() > 0.5 else -1.0
                grasper = 1 if self.grasped_wounded_persons() else 0
                return {"forward": fwd, "lateral": lat, "rotation": 0.0, "grasper": grasper}

        # ================= STATE MACHINE =================

        # --- EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0: self.blacklist_timer -= 1
            else: self.blacklisted_targets = []

            best_victim_pos = self.nav.victim_map.get_highest_score_target(obstacle_map=self.nav.obstacle_map)

            # [FIX] LỌC BỎ NẠN NHÂN TRONG VÙNG RETURN / RESCUE CENTER
            if best_victim_pos is not None and self.rescue_center_pos is not None:
                dist_to_home = np.linalg.norm(best_victim_pos - self.rescue_center_pos)
                
                # Nếu vị trí nghi ngờ nằm trong bán kính 100px quanh Rescue Center
                if dist_to_home < 100.0:
                    # print(f"[{self.identifier}] 🧹 Ignored victim at HOME (Safe Zone). Clearing map...")
                    # Xóa ngay điểm nóng này trên bản đồ để không bị lặp lại
                    self.nav.victim_map.clear_area(best_victim_pos, radius_grid=5)
                    best_victim_pos = None # Coi như không thấy

            if best_victim_pos is not None:
                is_claimed = False
                for t in self.busy_targets:
                    if np.linalg.norm(best_victim_pos - t['pos']) < 20.0:
                        is_claimed = True
                        break
                
                if not is_claimed:
                    self.current_target = best_victim_pos
                    self.comms.broadcast_claim_target(self.current_target)
                    self.state = "RESCUING"
                    self.position_before_rescue = self.estimated_pos.copy()
                    print(f"[{self.identifier}] 🚑 START RESCUE at {best_victim_pos}")

            if self.state == "EXPLORING": 
                if self.current_target is None:
                    # =========================================================
                    # TIER 1: FRONTIER XỊN (Có né đồng đội)
                    # =========================================================
                    frontier = self.nav.obstacle_map.get_frontier_target(
                        self.estimated_pos, 
                        self.estimated_angle,
                        busy_targets=self.busy_targets,
                        nearby_drones=nearby_drones_pos # [NEW] Truyền vào để né
                    )

                    if frontier is not None:
                        # Check blacklist
                        is_bad = False
                        for bad in self.blacklisted_targets:
                            if np.linalg.norm(frontier - bad) < 20.0: is_bad = True; break
                        
                        if not is_bad:
                            self.current_target = frontier
                            self.comms.broadcast_claim_target(self.current_target)
                            print(f"[{self.identifier}] 🎯 Tier 1 Target: Frontier")
                        else:
                            frontier = None # Bị blacklist thì coi như không tìm thấy

                    # =========================================================
                    # TIER 2: FALLBACK (Vùng tối bất kỳ)
                    # =========================================================
                    if self.current_target is None:
                        # Nếu Frontier xịn không có (hoặc bị blacklist), tìm đại vùng tối nào đó
                        unknown_target = self.nav.obstacle_map.get_unknown_target(
                            self.estimated_pos,
                            nearby_drones=nearby_drones_pos # Truyền vào để tìm chỗ vắng
                        )
                        
                        if unknown_target is not None:
                            self.current_target = unknown_target
                            self.comms.broadcast_claim_target(self.current_target)
                            print(f"[{self.identifier}] 🌑 Tier 2 Target: Unknown Area")

                    # =========================================================
                    # TIER 3: DESPERATION (Đi dạo ngẫu nhiên)
                    # =========================================================
                    if self.current_target is None:
                        random_free = self.nav.obstacle_map.get_random_free_target(self.estimated_pos)
                        if random_free is not None:
                            print(f"[{self.identifier}] 🎲 Tier 3 Target: Random Walk")
                            self.current_target = random_free
                        else:
                            # Hết cách: Xoay tại chỗ
                            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.8, "grasper": 0}
                
                elif best_victim_pos is None and self.current_target is not None:
                     if self.cnt_timestep % 5 == 0:
                         t_gx, t_gy = self.nav.obstacle_map.world_to_grid(self.current_target[0], self.current_target[1])
                         radius_check = 3
                         y_min = max(0, t_gy - radius_check)
                         y_max = min(self.nav.obstacle_map.grid_h, t_gy + radius_check + 1)
                         x_min = max(0, t_gx - radius_check)
                         x_max = min(self.nav.obstacle_map.grid_w, t_gx + radius_check + 1)
                         
                         sub_grid = self.nav.obstacle_map.grid[y_min:y_max, x_min:x_max]
                         if np.max(sub_grid) > 20.0:
                             print(f"[{self.identifier}] 🧱 TARGET TOO CLOSE TO WALL! Dropping...")
                             self.current_target = None
                             self.nav.current_astar_path = [] 

        # --- RESCUING ---
        elif self.state == "RESCUING":
            # Tăng timeout
            dist_to_target = 9999
            if self.current_target is not None:
                dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)
            
            # Chỉ đếm giờ nếu đang loay hoay gần mục tiêu (để tránh timeout khi đang bay từ xa tới)
            if dist_to_target < 100: 
                self.rescue_time += 1
            
            # Broadcast quyền sở hữu
            if self.cnt_timestep % 10 == 0 and self.current_target is not None:
                 self.comms.broadcast_claim_target(self.current_target)

            # 1. THU THẬP & TÍNH TOÁN RAY WALKING (Tìm điểm an toàn)
            visible_victims = []
            semantic_data = self.semantic_values()
            
            if semantic_data:
                for data in semantic_data:
                    if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                        angle_global = self.estimated_angle + data.angle
                        
                        # --- RAY WALKING (Giữ nguyên logic dò điểm an toàn) ---
                        best_safe_dist = 0.0 
                        check_range = np.arange(max(0.0, data.distance - 20.0), 0, -10.0)
                        found_valid_spot = False
                        for d in check_range:
                            cx = self.estimated_pos[0] + d * math.cos(angle_global)
                            cy = self.estimated_pos[1] + d * math.sin(angle_global)
                            if self.nav.obstacle_map.get_cost_at(np.array([cx, cy])) < 200.0:
                                best_safe_dist = d
                                found_valid_spot = True
                                break
                        if not found_valid_spot: best_safe_dist = 20.0 

                        safe_vx = self.estimated_pos[0] + best_safe_dist * math.cos(angle_global)
                        safe_vy = self.estimated_pos[1] + best_safe_dist * math.sin(angle_global)
                        
                        real_rx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                        real_ry = self.estimated_pos[1] + data.distance * math.sin(angle_global)

                        visible_victims.append({
                            'safe_pos': np.array([safe_vx, safe_vy]), 
                            'real_pos': np.array([real_rx, real_ry]), 
                            'dist': data.distance
                        })

            # 2. LOGIC "CHUNG TÌNH" (STUBBORN LOCKING)
            if visible_victims:
                # TRƯỜNG HỢP 1: Chưa có mục tiêu (Mới vào RESCUING)
                if self.last_rescue_pos is None:
                    # Chọn con gần nhất làm "Mối tình đầu"
                    challenger = min(visible_victims, key=lambda x: x['dist'])
                    self.current_target = challenger['safe_pos']
                    self.last_rescue_pos = challenger['real_pos'] # Khóa vị trí thực
                    # print(f"[{self.identifier}] 🔒 LOCKED on victim at {self.last_rescue_pos}")
                
                # TRƯỜNG HỢP 2: Đã có mục tiêu -> Chỉ update nếu NHÌN THẤY ĐÚNG CON ĐÓ
                else:
                    # Tìm trong đám đông xem con nào gần vị trí Last Rescue Pos nhất
                    tracker = min(visible_victims, key=lambda x: np.linalg.norm(x['real_pos'] - self.last_rescue_pos))
                    dist_track = np.linalg.norm(tracker['real_pos'] - self.last_rescue_pos)
                    
                    # Nếu sai số < 100cm -> Đúng là nó rồi -> Update vị trí cho chính xác hơn
                    if dist_track < 100.0:
                        self.current_target = tracker['safe_pos']
                        self.last_rescue_pos = tracker['real_pos']
                    
                    # [QUAN TRỌNG] KHÔNG CÓ ELSE!
                    # Nếu không tìm thấy (dist_track lớn), nghĩa là bị che khuất hoặc quay mặt đi.
                    # MẶC KỆ! Vẫn giữ current_target cũ và bay tới đó (Blind Approach).
                    # Tuyệt đối không switch sang con khác.

            # 3. TIMEOUT HANDLING (Chỉ bỏ cuộc khi hết giờ)
            if self.rescue_time >= 200:
                # print(f"[{self.identifier}] ⌛ RESCUE TIMEOUT! Abandoning target.")
                clear_pos = self.last_rescue_pos if self.last_rescue_pos is not None else self.current_target
                
                if clear_pos is not None:
                     self.nav.victim_map.clear_area(clear_pos, radius_grid=5)
                
                self.current_target = None
                self.last_rescue_pos = None
                self.state = 'EXPLORING'
                self.nav.current_astar_path = []
                self.rescue_time = 0
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            
            # 4. GRASP HANDLING
            if self.grasped_wounded_persons():
                self.rescue_time = 0
                self.state = "RETURNING"
                
                if self.last_rescue_pos is not None:
                    self.comms.broadcast_clear_zone(self.last_rescue_pos)
                    self.nav.victim_map.clear_area(self.last_rescue_pos, radius_grid=5)
                elif self.current_target is not None:
                     self.nav.victim_map.clear_area(self.current_target, radius_grid=5)
                
                self.current_target = self.initial_position 
                if self.current_target is None and self.rescue_center_pos is not None:
                    self.current_target = self.rescue_center_pos
                
                self.last_rescue_pos = None 
                print(f"[{self.identifier}] ✅ GRASPED! Returning home.")

        # --- RETURNING ---
        elif self.state == "RETURNING":
            if self.current_target is None:
                self.current_target = self.initial_position if self.initial_position is not None else self.rescue_center_pos

            if np.linalg.norm(self.estimated_pos - self.current_target) < 50.0 and steps_remaining > RETURN_TRIGGER_STEPS:
                self.state = "DROPPING"

        # --- DROPPING ---
        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            self.drop_step += 1
            if self.drop_step > 150 or not self.grasped_wounded_persons(): 
                print(f"[{self.identifier}] ⏬ DROPPED! Back to work.")
                self.state = "EXPLORING" 
                self.current_target = None
                self.drop_step = 0
                self.nav.current_astar_path = []
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return self.pilot.move_to_target_PID()

        # --- END GAME ---
        elif self.state == "END_GAME":
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # ================= EXECUTION =================
        next_waypoint = None
        if self.wait_scan_map == 0 and self.current_target is not None:
            next_waypoint = self.nav.get_next_waypoint(self.current_target)
            # print(f'Next way point is {next_waypoint}, dist {np.linalg.norm(self.estimated_pos - next_waypoint)}')
        
        dist_to_target = 9999.0
        if self.current_target is not None:
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)

        path_is_empty = (len(self.nav.current_astar_path) == 0)

        # 1. EXPLORING Logic
        if self.state == "EXPLORING" and self.current_target is not None:
            if dist_to_target < 40.0:
                self.nav.current_astar_path = []
                if self.wait_scan_map == 0:
                    print(f"[{self.identifier}] 🎯 ARRIVED at Frontier! Scanning...")
                
                self.patience = 0
                self.wait_scan_map += 1
                if self.wait_scan_map >= 20:
                    self.wait_scan_map = 0
                    self.current_target = None 
                
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0, "grasper": 0}
            else:
                self.wait_scan_map = 0

        # 2. RETURNING / RESCUING Logic (Fallback)
        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
            if path_is_empty and dist_to_target > 40.0:
                self.patience += 1
                if self.patience > 50:
                    print(f"[{self.identifier}] ⚠️ PATH STUCK. Wiggling/Fallback.")
                    self.patience = 0
                    
                    if self.state == "RETURNING" and self.initial_position is not None:
                         if np.linalg.norm(self.current_target - self.initial_position) > 10.0:
                             print(f"[{self.identifier}] 🔄 Switching to Initial Position for safety.")
                             self.current_target = self.initial_position
                             return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}

                    return {"forward": 0.0, "lateral": 1.0, "rotation": 0.5, "grasper": 1 if self.grasped_wounded_persons() else 0}

        if next_waypoint is None:
             return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1 if self.grasped_wounded_persons() else 0}
        
        real_target = self.current_target 
        self.current_target = next_waypoint 
        
        command = self.pilot.move_to_target_PID()
        
        self.current_target = real_target 
        
        return command

    def define_message_for_all(self):
        msg = self.outgoing_msg_buffer
        self.outgoing_msg_buffer = None 
        return msg