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
        self.state = "EXPLORING"
        self.last_state = ""
        
        self.current_target = None 
        self.rescue_center_pos = None 
        self.position_before_rescue = None
        self.initial_position = None
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

        # Danh sách đen
        self.blacklisted_targets = [] 
        self.blacklist_timer = 0
        
        # Logic chờ
        self.patience = 0
        self.wait_scan_map = 0

    def control(self) -> CommandsDict:
        self.cnt_timestep += 1
        
        # [LOG] State Change
        if self.state != self.last_state:
            print(f"[{self.identifier}] 🔄 STATE CHANGE: {self.last_state} -> {self.state}")
            self.last_state = self.state

        # 1. Update Navigator
        self.nav.update_navigator()
        
        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()
            print(f"[{self.identifier}] 🏁 STARTED at {self.initial_position}")

        # 2. Check Rescue Center
        semantic_data = self.semantic_values()
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    angle_global = self.estimated_angle + data.angle
                    obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    if self.rescue_center_pos is None:
                       self.rescue_center_pos = np.array([obj_x, obj_y])
                       # print(f"[{self.identifier}] 🏠 FOUND RESCUE CENTER at {self.rescue_center_pos}")

        # 3. Handle Communication
        self.comms.process_incoming_messages()
        self.comms.broadcast_map_updates()

        # [VISUALIZATION]
        if self.cnt_timestep % 5 == 0:
            self.nav.obstacle_map.display(
                self.estimated_pos, 
                current_target=self.current_target,
                current_path=self.nav.current_astar_path, 
                window_name=f"Obstacle Map - Drone {self.identifier}"
            )

        # [WARM-UP]
        if self.cnt_timestep < 50:
            if self.cnt_timestep == 1: print(f"[{self.identifier}] ⏳ WARMING UP...")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.5, "grasper": 0}

        # 4. Check Health / Return
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

        # Anti-stuck
        self.pos_history_long.append(self.estimated_pos.copy())
        waiting = 100 if self.state == 'EXPLORING' else 130
        if len(self.pos_history_long) > waiting: self.pos_history_long.pop(0) 
        
        if self.state != "END_GAME" and len(self.pos_history_long) == waiting and steps_remaining > RETURN_TRIGGER_STEPS:
            start_pos = self.pos_history_long[0]
            dist_moved = np.linalg.norm(self.estimated_pos - start_pos)
            if dist_moved < 8.0:
                print(f"[{self.identifier}] ⚠️ STUCK DETECTED! Wiggling...")
                self.patience = 0 
                fwd = 0
                lat = 1.0 if random.random() > 0.5 else -1.0
                grasper = 1 if self.grasped_wounded_persons() else 0
                return {"forward": fwd, "lateral": lat, "rotation": 0.0, "grasper": grasper}

        # ================= STATE MACHINE =================

        # --- EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0:
                self.blacklist_timer -= 1
            else:
                self.blacklisted_targets = [] 

            # [FIX 1] ƯU TIÊN KIỂM TRA NẠN NHÂN TRƯỚC (Override Frontier)
            best_victim_pos = self.nav.victim_map.get_highest_score_target()
            
            # Nếu tìm thấy nạn nhân, GHI ĐÈ target ngay lập tức
            if best_victim_pos is not None:
                dist_to_victim = np.linalg.norm(self.estimated_pos - best_victim_pos)
                
                # Update target thành nạn nhân
                self.current_target = best_victim_pos
                
                # Nếu đã đến gần -> Chuyển sang RESCUING
                if dist_to_victim < 30.0:
                    self.state = "RESCUING"
                    self.position_before_rescue = self.estimated_pos.copy()
                    print(f"[{self.identifier}] 🚑 START RESCUE at {best_victim_pos}")
                else:
                    # Vẫn ở EXPLORING nhưng hướng về nạn nhân
                    # (Code sẽ chạy xuống phần EXECUTION để đi đến đây)
                    pass

            # [FIX 2] Chỉ tìm Frontier MỚI nếu KHÔNG CÓ nạn nhân VÀ KHÔNG CÓ target
            elif self.current_target is None:
                frontier = self.nav.obstacle_map.get_frontier_target(
                    self.estimated_pos, 
                    self.estimated_angle
                )
                
                if frontier is not None:
                    is_bad = False
                    for bad in self.blacklisted_targets:
                        if np.linalg.norm(frontier - bad) < 20.0:
                            is_bad = True
                            break
                    
                    if not is_bad:
                        self.current_target = frontier
                    else:
                        # Random Walk
                        angle = random.uniform(-math.pi, math.pi)
                        dist = 100
                        rx = self.estimated_pos[0] + dist * math.cos(angle)
                        ry = self.estimated_pos[1] + dist * math.sin(angle)
                        self.current_target = np.array([rx, ry])
                else:
                    print(f"[{self.identifier}] 😵 NO FRONTIER! Spinning...")
                    return {"forward": 0.0, "lateral": 0.0, "rotation": 0.8, "grasper": 0}
            
            # Check nếu Frontier hiện tại bị biến thành tường (nhưng không phải nạn nhân)
            elif best_victim_pos is None and self.current_target is not None:
                 t_gx, t_gy = self.nav.obstacle_map.world_to_grid(self.current_target[0], self.current_target[1])
                 if self.nav.obstacle_map.grid[t_gy, t_gx] > 20.0:
                     print(f"[{self.identifier}] 🧱 TARGET IS WALL! Finding new...")
                     self.current_target = None

        # --- RESCUING ---
        elif self.state == "RESCUING":
            # Tracking chính xác
            closest_victim_now = None
            min_dist = float('inf')
            semantic_data = self.semantic_values()
            if semantic_data:
                for data in semantic_data:
                    if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                        angle_global = self.estimated_angle + data.angle
                        vx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                        vy = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                        if data.distance < min_dist:
                            min_dist = data.distance
                            closest_victim_now = np.array([vx, vy])

            if closest_victim_now is not None:
                self.current_target = closest_victim_now
            
            if self.grasped_wounded_persons():
                self.state = "RETURNING"
                self.comms.broadcast_clear_zone(self.current_target)
                
                # Clear map local
                gx, gy = self.nav.victim_map.world_to_grid(self.current_target[0], self.current_target[1])
                self.nav.victim_map.grid[gy-2:gy+3, gx-2:gx+3] = 0.0
                
                # [FIX 3] Khi về, ưu tiên INITIAL_POSITION vì nó an toàn
                self.current_target = self.initial_position 
                if self.current_target is None and self.rescue_center_pos is not None:
                    self.current_target = self.rescue_center_pos

                print(f"[{self.identifier}] ✅ GRASPED! Returning home.")

        # --- RETURNING ---
        elif self.state == "RETURNING":
            # Đảm bảo target luôn có
            if self.current_target is None:
                self.current_target = self.initial_position if self.initial_position is not None else self.rescue_center_pos

            # Check về đích
            if np.linalg.norm(self.estimated_pos - self.current_target) < 20.0:
                self.state = "DROPPING"

        # --- DROPPING ---
        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            self.drop_step += 1
            if self.drop_step > 150: 
                print(f"[{self.identifier}] ⏬ DROPPED! Back to work.")
                self.state = "EXPLORING" 
                self.current_target = None
                self.drop_step = 0
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return self.pilot.move_to_target_PID()

        # --- END GAME ---
        elif self.state == "END_GAME":
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # ================= EXECUTION =================
        next_waypoint = None
        if self.wait_scan_map == 0 and self.current_target is not None:
            next_waypoint = self.nav.get_next_waypoint(self.current_target)
        
        dist_to_target = 9999.0
        if self.current_target is not None:
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)

        path_is_empty = (len(self.nav.current_astar_path) == 0)

        # 1. EXPLORING Logic
        if self.state == "EXPLORING" and self.current_target is not None:
            if path_is_empty:
                # Đã đến nơi -> Scan
                if dist_to_target < 30.0:
                    if self.wait_scan_map == 0:
                        print(f"[{self.identifier}] 🎯 ARRIVED at Frontier! Scanning...")
                    
                    self.patience = 0
                    self.wait_scan_map += 1
                    if self.wait_scan_map > 40:
                        self.wait_scan_map = 0
                        self.current_target = None 
                    
                    return {"forward": 0.0, "lateral": 0.0, "rotation": 0, "grasper": 0}
                
                # Bị chặn -> Blacklist
                else:
                    self.patience += 1
                    self.wait_scan_map = 0
                    if self.patience > 80:
                        self.blacklisted_targets.append(self.current_target.copy())
                        self.blacklist_timer = 50 
                        print(f"[{self.identifier}] 🚫 PATH FAILED -> BLACKLISTED")
                        self.current_target = None
                        self.patience = 0
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 1.0, "grasper": 0}
                    else:
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # 2. RETURNING / RESCUING Logic (Fallback)
        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
            if path_is_empty and dist_to_target > 40.0:
                self.patience += 1
                if self.patience > 50:
                    print(f"[{self.identifier}] ⚠️ PATH STUCK. Wiggling/Fallback.")
                    self.patience = 0
                    
                    # [FIX 4] Nếu đang về nhà mà kẹt đường -> Đổi sang Initial Position (nếu đang dùng RescueCenter)
                    if self.state == "RETURNING" and self.initial_position is not None:
                         # Nếu đang nhắm RescueCenter mà fail -> Về Initial Position
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