import math
import random 
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


from swarm_rescue.solutions.navigator import Navigator
from swarm_rescue.solutions.pilot import Pilot
from swarm_rescue.solutions.communicator import CommunicatorHandler
from swarm_rescue.solutions.victim_manager import VictimManager





class MyStatefulDrone(DroneAbstract):
    """
    Main Drone Driver Class.
    Implements a state machine (DISPERSING -> EXPLORING <-> RESCUING -> RETURNING).
    Integrates Navigator (Planning), Pilot (Control), and VictimManager (Data).
    """
    
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
        self.victim_manager = VictimManager(self)
        
        # --- STATE VARIABLES ---
        self.state = "DISPERSING" 
        self.current_target = None 
        self.rescue_center_pos = None 
        self.initial_position = None 
        self.cnt_timestep = 0
        self.current_target_best_victim_pos = None
        
        self.last_rescue_pos = None
        self.drop_step = 0
        self.rescue_time = 0
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        
        # Anti-stuck & Logic Variables
        self.pos_history_long = []
        self.patience = 0
        self.blacklisted_targets = []
        self.blacklist_timer = 0
        self.rescue_time = 0
        
        # Counts consecutive pathfinding failures to implement "Patience" before blacklisting
        self.path_fail_count = 0 
        self.STUCK_TIME_EXPLORING = 50 #(timestep)
        self.STUCK_TIME_OTHER = 70 #(timestep)
        self.MAX_SPEED = 0.9 #in [0,1]
        self.RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        self.floodfill_cooldown = 0
        # [NEW] Counter for map completion
        self.no_frontier_patience = 0
        self.preferred_angle = None
        self.safe_dispersion_reached_tick = None
        self.panic_timer = 0

    def draw_top_layer(self):
        self.draw_identifier()

    def control(self) -> CommandsDict:
        """
        Main control loop called by the simulator every step.
        """
        self.draw_top_layer()
        self.RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        
        self.cnt_timestep += 1
        
        # 1. SENSING
        semantic_data = self.semantic_values() 
        self.victim_manager.update_from_sensor(self.estimated_pos, self.estimated_angle, semantic_data, self.cnt_timestep)
        known_victims_pos = [r['pos'] for r in self.victim_manager.registry]
        nearby_drones_pos = []
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    angle_global = self.estimated_angle + data.angle
                    dx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    dy = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    nearby_drones_pos.append(np.array([dx, dy]))

        self.nav.update_navigator(nearby_drones=nearby_drones_pos, nearby_victims=known_victims_pos)

        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()
            print(f"[{self.identifier}] 🏁 STARTED.")

        if self.cnt_timestep % 100 == 0: print(self.cnt_timestep)
        # 3. Locate Rescue Center
        check_center = False
        if semantic_data:
            tmp = float('inf')
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    check_center = True
                    angle_global = self.estimated_angle + data.angle
                    obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    dist_to_center = np.linalg.norm(self.estimated_pos - np.array([obj_x, obj_y]))
                    if dist_to_center < tmp:
                        tmp = dist_to_center
                        self.rescue_center_pos = np.array([obj_x, obj_y])

        # Debug visualization
        if self.cnt_timestep % 5 == 0:
            self.nav.obstacle_map.display(
                self.estimated_pos, 
                current_target=self.current_target,
                current_path=self.nav.current_astar_path, 
                window_name=f"Map - Drone {self.identifier}"
            )

        # 4. Receive and process messages

        self.comms.process_incoming_messages()

        # ================= STATE MACHINE =================

        # --- STATE: DISPERSING ---
        if self.state == "DISPERSING":
            # Initialize a memory variable to track when safe distance is reached
                
            dx = self.estimated_pos[0] - self.initial_position[0]
            dy = self.estimated_pos[1] - self.initial_position[1]
            dist_moved = math.hypot(dx, dy)
            
            # Check if we have fulfilled the escape conditions
            if self.safe_dispersion_reached_tick is None:
                # Condition: Run for at least 50 ticks AND exceed 150px safe distance.
                # Safety Timeout: Force escape at 200 ticks to prevent infinite deadlock.
                if (self.cnt_timestep >= 50 and dist_moved >= 150.0) or self.cnt_timestep >= 1:
                    self.safe_dispersion_reached_tick = self.cnt_timestep
            
            # 1. Active Repulsion: Push away from peers, walls, and Rescue Center
            if self.safe_dispersion_reached_tick is None:
                #print(f'{self.identifier} {self.state} move_function 1')
                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)

            
            # 2. Scanning Mode: Spin to map surroundings (Rescue Center) with Lidar
            elif self.cnt_timestep < self.safe_dispersion_reached_tick + 50: 
                #print(f'{self.identifier} {self.state} move_function 21')
                return self.pilot.move_function(forward=0, lateral=0, rotation=1, grasper=0, repulsive_force_bool=True)
            
            # 3. Lock Natural Escape Angle & Transition to Exploring
            else:
                if dist_moved > 5.0:
                    self.preferred_angle = math.atan2(dy, dx)
                else: 
                    self.preferred_angle = self.estimated_angle
                    
                deg_angle = math.degrees(self.preferred_angle)
                print(f"[{self.identifier}] 🚀 WARMUP DONE. Dist: {dist_moved:.1f}px. Angle: {deg_angle:.1f}°. EXPLORING!")
                self.state = "EXPLORING"
        # --- BATTERY GUARD ---
        self.steps_remaining = self.max_timesteps - self.cnt_timestep
        self.pilot.low_battery()

        # --- ANTI-STUCK & PANIC MODE ---
        is_currently_stuck = self.nav.is_stuck()
            
        if self.state != "END_GAME":
            # Kích hoạt bộ đếm Panic nếu bị kẹt hoặc tuyệt vọng vì hết map
            if is_currently_stuck:
                self.panic_timer = 80 # Cấp 80 tick để lách qua kẹt
            elif self.state == "EXPLORING" and self.no_frontier_patience > 10:
                self.panic_timer = 80
                self.no_frontier_patience = 0 # Tránh cộng dồn báo động giả
                
        # Đồng bộ trạng thái hoảng loạn xuống hệ thống Mapping
        if self.panic_timer > 0:
            self.panic_timer -= 1
            if not getattr(self.nav.obstacle_map, 'panic_mode', False):
                self.nav.obstacle_map.panic_mode = True
                self.nav.obstacle_map.update_cost_map() # [QUAN TRỌNG] Ép tạo lại map tức thì!
                print(f"[{self.identifier}] 🚨 PANIC MODE ON: Flattening Cost Map to escape narrow corridor!")
        else:
            if getattr(self.nav.obstacle_map, 'panic_mode', False):
                self.nav.obstacle_map.panic_mode = False
                self.nav.obstacle_map.update_cost_map() # Khôi phục lại bản đồ an toàn
                print(f"[{self.identifier}] 😌 Panic Mode OFF: Normal navigation resumed.")

        # Xử lý giãy giụa khi kẹt
        if is_currently_stuck:
            self.nav.current_astar_path = []
            
            # Wall Repulsion Unstuck
            grasper = 1 if self.grasped_wounded_persons() else 0
            #print(f'{self.identifier} {self.state} move_function 2')
            return self.pilot.move_function(forward = 0, lateral = 0, rotation = 1.0, grasper = grasper, repulsive_force_bool = True)


        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0: self.blacklist_timer -= 1
            else: self.blacklisted_targets = []

            # 1. Check for Victims (Highest Priority)
            best_victim_pos = self.victim_manager.get_nearest_victim(self.estimated_pos, self.blacklisted_targets)
            # Ignore victims at home base (already rescued)
            # if best_victim_pos is not None and self.rescue_center_pos is not None:
            #     if np.linalg.norm(best_victim_pos - self.rescue_center_pos) < 200.0:
            #         self.victim_manager.delete_victim_at(best_victim_pos)
            #         best_victim_pos = None 

            if best_victim_pos is not None:
                self.current_target = best_victim_pos
                self.state = "RESCUING"
                self.path_fail_count = 0 
                print(f"[{self.identifier}] 🚑 RESCUING VICTIM at {best_victim_pos}")

            # 2. Find Frontier (Exploration)
            if self.state == "EXPLORING": 
                if self.floodfill_cooldown > 0:
                    self.floodfill_cooldown -= 1
                
                if self.current_target is None and self.floodfill_cooldown <= 0:
                    # You can pass self.identifier here later if you implement Directional Bias
                    frontier, path = self.nav.obstacle_map.get_reachable_frontier_and_path(
                        self.estimated_pos, 
                        self.estimated_angle, 
                        self.preferred_angle,
                        self.initial_position,
                        self.rescue_center_pos
                    )
                    
                    if frontier is not None:
                        is_bad = False
                        for bad in self.blacklisted_targets:
                            if np.linalg.norm(frontier - bad) < 20.0: is_bad = True; break
                        
                        if not is_bad:
                            self.current_target = frontier
                            self.nav.current_astar_path = path 
                            self.nav.last_path_index = 0
                            self.nav.last_astar_target = frontier.copy() 
                            self.path_fail_count = 0
                            # Reset patience when a frontier is found
                            self.no_frontier_patience = 0
                        else: self.current_target = None
                    else:
                        self.current_target = None
                        self.floodfill_cooldown = 15
                    
                    # [NEW] Fallback: Lost in the fog or Map fully explored
                    if self.current_target is None:
                        self.no_frontier_patience += 1
                        
                        # Increased patience to 40 ticks to give it time to spin and scan
                        if self.no_frontier_patience >= 400:
                            print(f"[{self.identifier}] 🌍 MAP FULLY EXPLORED! Returning to base.")
                            self.state = "RETURNING"
                            self.nav.current_astar_path = []
                            self.current_target = self.initial_position if self.initial_position is not None else self.rescue_center_pos
                            #print(f'{self.identifier} {self.state} move_function 3')
                            return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=0, repulsive_force_bool= True)
                            
                        # "Struggle" maneuver: Move slightly and spin hard to force Lidar to clear local 'Unknown' fog
                        #print(f'{self.identifier} {self.state} move_function 4')
                        return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0.8, grasper = 0, repulsive_force_bool = True)

        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":
            dist_to_target = 9999
            if self.current_target is not None:
                dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)
            if dist_to_target < 70: self.rescue_time += 1

            if semantic_data:
                for data in semantic_data:
                    # if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    #     print(f'Debug type: {data.entity_type}, grasped: {data.grasped}!')
                    if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                        angle_global = self.estimated_angle + data.angle
                        best_safe_dist = 20.0 
                        check_range = np.arange(max(0.0, data.distance - 20.0), 0, -10.0)
                        for d in check_range:
                            cx = self.estimated_pos[0] + d * math.cos(angle_global)
                            cy = self.estimated_pos[1] + d * math.sin(angle_global)
                            if self.nav.obstacle_map.get_cost_at(np.array([cx, cy])) < 300.0:
                                best_safe_dist = d; break
                        safe_vx = self.estimated_pos[0] + best_safe_dist * math.cos(angle_global)
                        safe_vy = self.estimated_pos[1] + best_safe_dist * math.sin(angle_global)
                        self.current_target = np.array([safe_vx, safe_vy])
                        break 

            # Rescue Timeout
            if self.rescue_time >= 50:
                if self.current_target is not None: 
                    print(f"{[self.identifier]} Delete victim at {self.current_target}")
                    self.victim_manager.delete_victim_at(self.current_target)
                self.current_target = None
                self.state = 'EXPLORING'; self.nav.current_astar_path = []; self.rescue_time = 0
                #print(f'{self.identifier} {self.state} move_function 5')
                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)
            
            # Successful Grasp
            if self.grasped_wounded_persons():
                self.rescue_time = 0; self.state = "RETURNING"
                self.victim_manager.delete_victim_at(self.estimated_pos)
                self.current_target = self.initial_position
                if self.current_target is None and self.rescue_center_pos is not None:
                    self.current_target = self.rescue_center_pos
                print(f"[{self.identifier}] ✅ GRASPED! Returning.")

        # --- RETURNING & DROPPING ---
        elif self.state == "RETURNING":
            if self.current_target is None: self.current_target = self.initial_position
            if np.linalg.norm(self.estimated_pos - self.current_target) < 50.0:
                self.state = "DROPPING"

        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            if check_center: self.drop_step += 1
            
            if self.drop_step > 150 or not self.grasped_wounded_persons(): 
                self.drop_step = 0
                self.nav.current_astar_path = []
                self.current_target = None
                
                # Check if it returns home because it's done full map
                if self.no_frontier_patience >= 3:
                    print(f"[{self.identifier}] 🛌 MAP DONE. Resting at base.")
                    self.state = "END_GAME" # Rest
                else:
                    print(f"[{self.identifier}] ⏬ DROPPED. Resume Exploring.")
                    self.state = "EXPLORING" # Continue exploring
                    
                #print(f'{self.identifier} {self.state} move_function 6')
                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)              
            #print(f'{self.identifier} {self.state} move_to_target_carrot 7')
            return self.pilot.move_to_target_carrot()

        # --- STATE: END GAME ---
        elif self.state == "END_GAME":
            if not self.is_inside_return_area:
                self.state = "RETURNING"
            #print(f'{self.identifier} {self.state} move_function 8')    
            return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)


        # ================= EXECUTION =================
        next_waypoint = None

        if self.current_target is not None:
            dist = np.linalg.norm(self.estimated_pos - self.current_target)
            if len(self.nav.current_astar_path) == 0 and dist > 40.0:
                self.path_fail_count += 1 
                if self.path_fail_count > 30:
                    print(f"[{self.identifier}] ❌ Path failed.")
                    
                    # Only blaclist if not returning
                    if self.state != "RETURNING":
                        print("   -> Blacklisting target.")
                        self.blacklisted_targets.append(self.current_target)
                    
                    self.blacklist_timer = 200
                    self.current_target = None
                    self.path_fail_count = 0
                    
                    # Check grasper
                    keep_grasping = 1 if self.grasped_wounded_persons() else 0
                    
                    # Turn to unstuck
                    #print(f'{self.identifier} {self.state} move_function 9')
                    return self.pilot.move_function(forward = 0, lateral = 0, rotation = 1.0, grasper = keep_grasping, repulsive_force_bool = True)
            else:
                self.path_fail_count = 0
            
            # Navigator will return None if path becomes blocked
            next_waypoint = self.nav.get_next_waypoint(self.current_target)

        # Fallback Check
        dist_to_target = 9999.0
        if self.current_target is not None:
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)

        # [UPDATED] Robust Arrival Check
        if self.state == "EXPLORING" and self.current_target is not None:
            # 1. Check if we arrived close enough
            arrived_close = (dist_to_target < 35.0)
            
            # 2. Get cell value at the target position
            gx, gy = self.nav.obstacle_map.world_to_grid(self.current_target[0], self.current_target[1])
            cell_value = 0.0
            if 0 <= gx < self.nav.obstacle_map.grid_w and 0 <= gy < self.nav.obstacle_map.grid_h:
                cell_value = self.nav.obstacle_map.grid[gy, gx]
            
            # Obstacle check (Wall appeared)
            target_obstructed = (cell_value > 10.0)
            
            # [NEW] 2.5. Check for Stale Target (Area already explored by teammates)
            # Instead of checking just the target cell (which is always FREE),
            # we check a 5x5 window around the target.
            # If there are NO unknown cells left in this window, the frontier is stale.
            target_stale = True
            search_radius = 2 # 5x5 window
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= nx < self.nav.obstacle_map.grid_w and 0 <= ny < self.nav.obstacle_map.grid_h:
                        val = self.nav.obstacle_map.grid[ny, nx]
                        # If we find at least one UNKNOWN cell (-5.0 to 5.0), it's still a valid frontier
                        if -0.1 < val < 5.0:
                            target_stale = False
                            break
                if not target_stale:
                    break

            if target_stale and dist_to_target < 150.0:
                target_stale = False

            # Trigger reset if any condition is met
            if arrived_close or target_obstructed or target_stale:
                # print(f"[{self.identifier}] Target Done/Invalid/Stale (Obs:{target_obstructed}, Stale:{target_stale}). Resetting.")
                self.current_target = None
                self.nav.current_astar_path = []
                #print(f'{self.identifier} {self.state} move_function 10')
                if self.grasped_wounded_persons(): grasper = 1
                else: grasper = 0

                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = grasper, repulsive_force_bool = True)

        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
             if next_waypoint is None:
                # Path blocked in rescue mode -> Replan A* will trigger in next loop by Nav
                # But here we handle stuck patience
                self.patience += 1
                if self.patience > 50:
                    self.patience = 0
                    if self.state == "RETURNING" and self.initial_position is not None:
                         if np.linalg.norm(self.current_target - self.initial_position) > 10.0:
                             self.current_target = self.initial_position
                             print(f'{self.identifier} {self.state} move_function 11')
                             return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 1, repulsive_force_bool = True)
                    if self.grasped_wounded_persons(): grasped = 1
                    else: grasped = 0
                    #print(f'{self.identifier} {self.state} move_function 12')
                    return self.pilot.move_function(forward = 0, lateral = 1, rotation = 0.5, grasper = grasped, repulsive_force_bool = True)

        # [NEW] FAST UNSTUCK & PATH FAILURE HANDLING
        if next_waypoint is None:
            grasper = 1 if self.grasped_wounded_persons() else 0
            
            # If Exploring, trigger cooldown so it doesn't instantly lag CPU with Flood Fill
            if self.state == "EXPLORING" and self.current_target is not None:
                self.floodfill_cooldown = 15 # Wait 15 ticks before thinking again
                self.current_target = None
                self.nav.current_astar_path = []
            
            # 1. Check Lidar for walls
            lidar_data = self.lidar_values()
            if lidar_data is not None:
                min_dist = min(lidar_data)
                
                # 2. If stuck near a wall (in the high cost inflation zone, < 60px)
                if min_dist < 60.0:
                    # Use Pilot's repulsive force (boosted to 0.8) to push away
                    print(f'[{self.identifier}] STUCK, trying to push from wall')
                    #print(f'{self.identifier} {self.state} move_function 13')
                    return self.pilot.move_function(forward= 0, lateral = 0, rotation= 0.8, grasper = int(grasper), repulsive_force_bool= True)
            
            # Fallback if no wall is nearby (stuck for other reasons)
            #print(f'{self.identifier} {self.state} move_function 14')
            return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = grasper, repulsive_force_bool = True)
        
        # Execute Pilot Command
        real_target = self.current_target 
        self.current_target = next_waypoint 
        self.MAX_SPEED = 1.0 if self.state == "RETURNING" else 0.9
        #print(f'{self.identifier} {self.state} move_to_target_carrot_function_call my_drone_driver')
        command = self.pilot.move_to_target_carrot()
        self.current_target = real_target 
        return command

    def define_message_for_all(self):
        
        return_dict = self.comms.create_new_message()
        return return_dict