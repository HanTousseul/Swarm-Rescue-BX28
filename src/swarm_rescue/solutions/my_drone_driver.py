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
        self.rescue_center_masked = False
        self.patience = 0
        self.blacklisted_targets = []
        self.blacklist_timer = 0
        self.rescue_time = 0
        
        # Counts consecutive pathfinding failures to implement "Patience" before blacklisting
        self.path_fail_count = 0 

        # CONSTANTS 
        self.STUCK_TIME_EXPLORING = 50 #(timestep)
        self.STUCK_TIME_OTHER = 70 #(timestep)
        self.MAX_SPEED = 0.9 #in [0,1]
        self.RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)

    def control(self) -> CommandsDict:
        """
        Main control loop called by the simulator every step.
        """
        
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

        # Mask Rescue Center on map to prevent exploring inside walls
        if not self.rescue_center_masked and self.rescue_center_pos is not None:
            self.nav.obstacle_map.mask_rescue_center(self.rescue_center_pos)
            self.rescue_center_masked = True
            print(f"[{self.identifier}] 🚫 Masked Rescue Center.")

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
        # Initial spread to avoid collisions at spawn.
        if self.state == "DISPERSING":
            if self.cnt_timestep < 60:
                
                forward, lateral = self.pilot.repulsive_force()
                return self.pilot.move_function(forward = forward, lateral = lateral, rotation = 0, grasper = 0, repulsive_force_bool = True)
            
            else:
                print(f"[{self.identifier}] 🚀 WARMUP DONE. EXPLORING!")
                self.state = "EXPLORING"

        # --- BATTERY GUARD ---
        self.steps_remaining = self.max_timesteps - self.cnt_timestep
        self.pilot.low_battery()

        # --- ANTI-STUCK ---
        # Checks if position hasn't changed significantly over a time window.
        if self.nav.is_stuck():
            self.nav.current_astar_path = []
            
            # [NEW STRATEGY] Wall Repulsion Unstuck
            # 1. Tính lực đẩy từ tường (dùng mode aggressive để lực mạnh hơn)
            radial,orthoradial = self.pilot.repulsive_force()
            
            grasper = 1 if self.grasped_wounded_persons() else 0
            return self.pilot.move_function(forward = radial, lateral = orthoradial, rotation = 0, grasper = grasper, repulsive_force_bool = True)

        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0: self.blacklist_timer -= 1
            else: self.blacklisted_targets = []

            # 1. Check for Victims (Highest Priority)
            best_victim_pos = self.victim_manager.get_nearest_victim(self.estimated_pos, self.blacklisted_targets)
            # Ignore victims at home base (already rescued)
            if best_victim_pos is not None and self.rescue_center_pos is not None:
                if np.linalg.norm(best_victim_pos - self.rescue_center_pos) < 100.0:
                    self.victim_manager.delete_victim_at(best_victim_pos)
                    best_victim_pos = None 

            if best_victim_pos is not None:
                self.current_target = best_victim_pos
                self.state = "RESCUING"
                self.path_fail_count = 0 
                print(f"[{self.identifier}] 🚑 RESCUING VICTIM at {best_victim_pos}")

            # 2. Find Frontier (Exploration)
            if self.state == "EXPLORING": 
                # If current target is None (Arrived, Blocked, or Started)
                if self.current_target is None:
                    frontier, path = self.nav.obstacle_map.get_reachable_frontier_and_path(self.estimated_pos, self.estimated_angle)
                    
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
                        else: self.current_target = None
                    
                    if self.current_target is None:
                        self.current_target = self.nav.obstacle_map.get_unknown_target(self.estimated_pos, nearby_drones=nearby_drones_pos)
                        self.path_fail_count = 0

                    if self.current_target is None:
                        self.current_target = self.nav.obstacle_map.get_random_free_target(self.estimated_pos)
                        self.path_fail_count = 0
                        if self.current_target is None:
                            return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0.8, grasper = 0, repulsive_force_bool = True)

        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":
            dist_to_target = 9999
            if self.current_target is not None:
                dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)
            if dist_to_target < 70: self.rescue_time += 1

            if semantic_data:
                for data in semantic_data:
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
            if self.rescue_time >= 100:
                if self.current_target is not None: 
                    print(f"{[self.identifier]} Delete victim at {self.current_target}")
                    self.victim_manager.delete_victim_at(self.current_target)
                self.current_target = None
                self.state = 'EXPLORING'; self.nav.current_astar_path = []; self.rescue_time = 0
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
            if np.linalg.norm(self.estimated_pos - self.current_target) < 50.0 and self.steps_remaining > self.RETURN_TRIGGER_STEPS:
                self.state = "DROPPING"

        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            if check_center: self.drop_step += 1
            if self.drop_step > 150 or not self.grasped_wounded_persons(): 
                print(f"[{self.identifier}] ⏬ DROPPED. Resume Exploring.")
                self.drop_step = 0
                self.nav.current_astar_path = []
                self.state = "EXPLORING" 
                self.current_target = None
                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)
            return self.pilot.move_to_target_carrot()

        # --- STATE: END GAME ---
        elif self.state == "END_GAME":
            self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 0, repulsive_force_bool = True)

        # ================= EXECUTION =================
        next_waypoint = None

        if self.current_target is not None:
            dist = np.linalg.norm(self.estimated_pos - self.current_target)
            
            USE_DIRECT_PID = False
            if (self.state in ["RESCUING", "RETURNING"]) and (dist < 150.0):
                if self.nav.obstacle_map.check_line_of_sight(self.estimated_pos, self.current_target, safety_radius=1, check_cost=False):
                    USE_DIRECT_PID = True
            
            if USE_DIRECT_PID:
                next_waypoint = self.current_target
                self.path_fail_count = 0
            else:
                if len(self.nav.current_astar_path) == 0 and dist > 40.0:
                    self.path_fail_count += 1 
                    if self.path_fail_count > 30:
                        print(f"[{self.identifier}] ❌ Path failed.")
                        
                        # [FIX 1] Chỉ Blacklist nếu KHÔNG PHẢI đang về nhà
                        # Nếu đang RETURNING, ta không muốn blacklist cái Home.
                        if self.state != "RETURNING":
                            print("   -> Blacklisting target.")
                            self.blacklisted_targets.append(self.current_target)
                        
                        self.blacklist_timer = 200
                        self.current_target = None
                        self.path_fail_count = 0
                        
                        # [FIX 2] Kiểm tra Grasper state để không làm rơi nạn nhân
                        keep_grasping = 1 if self.grasped_wounded_persons() else 0
                        
                        # Xoay người để thoát kẹt, nhưng vẫn giữ victim
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
            
            # 2. Check if the target point has become obstructed (Wall appeared)
            gx, gy = self.nav.obstacle_map.world_to_grid(self.current_target[0], self.current_target[1])
            cell_value = 0.0
            if 0 <= gx < self.nav.obstacle_map.grid_w and 0 <= gy < self.nav.obstacle_map.grid_h:
                cell_value = self.nav.obstacle_map.grid[gy, gx]
            target_obstructed = (cell_value > 10.0)
            
            # 3. Check if Navigator invalidated path (next_waypoint is None)
            path_invalid = (next_waypoint is None)

            if arrived_close or target_obstructed or path_invalid:
                # print(f"[{self.identifier}] Target Done/Invalid (Obs:{target_obstructed}). Resetting.")
                self.current_target = None
                self.nav.current_astar_path = []
                if self.grasped_wounded_persons(): grasper = 1
                else: grasper = 0
                return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = grasper, repulsive_force_bool = True)

        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
             if next_waypoint is None and not USE_DIRECT_PID:
                # Path blocked in rescue mode -> Replan A* will trigger in next loop by Nav
                # But here we handle stuck patience
                self.patience += 1
                if self.patience > 50:
                    self.patience = 0
                    if self.state == "RETURNING" and self.initial_position is not None:
                         if np.linalg.norm(self.current_target - self.initial_position) > 10.0:
                             self.current_target = self.initial_position
                             return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = 1, repulsive_force_bool = True)
                         
                    if self.grasped_wounded_persons(): grasped = 1
                    else: grasped = 0
                    return self.pilot.move_function(forward = 0, lateral = 1, rotation = 0.5, grasper = grasped)

        if next_waypoint is None:
            print(f'[{self.identifier}] No next waypoint {self.estimated_pos}')
            if self.grasped_wounded_persons(): grasper = 1
            else: grasper = 0
            
            return self.pilot.move_function(forward = 0, lateral = 0, rotation = 0, grasper = grasper, repulsive_force_bool = True)
        
        # Execute Pilot Command
        real_target = self.current_target 
        self.current_target = next_waypoint 
        command = self.pilot.move_to_target_carrot()
        self.current_target = real_target 
        return command

    def define_message_for_all(self):
        
        return_dict = self.comms.create_new_message()
        return return_dict