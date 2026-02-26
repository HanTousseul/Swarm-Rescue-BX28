import math
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
    Implements a Finite State Machine (FSM): DISPERSING -> EXPLORING <-> RESCUING -> RETURNING -> DROPPING.
    Acts as the central orchestrator, integrating Navigator (Pathfinding), Pilot (Locomotion), 
    and VictimManager (Data handling).
    """
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)

        # --- CONFIGURATION ---
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
        
        self.drop_step = 0
        self.rescue_time = 0
        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        
        # --- ANTI-STUCK & BEHAVIORAL LOGIC VARIABLES ---
        self.pos_history_long = []
        self.patience = 0
        self.blacklisted_targets = []
        self.blacklist_timer = 0
        
        self.path_fail_count = 0 
        self.STUCK_TIME_EXPLORING = 50 
        self.STUCK_TIME_OTHER = 70 
        self.MAX_SPEED = 0.9 
        self.RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        
        self.floodfill_cooldown = 0
        self.preferred_angle = None
        self.safe_dispersion_reached_tick = None
        self.panic_timer = 0
        self.current_target_best_victim_pos = None

    def control(self) -> CommandsDict:
        """
        The main control loop executed by the simulator at every timestep.
        Follows the Sense -> Plan -> Act architecture.
        """
        self.cnt_timestep += 1
        
        # ================= 1. SENSING & MAPPING =================
        semantic_data = self.semantic_values() 
        self.victim_manager.update_from_sensor(self.estimated_pos, self.estimated_angle, semantic_data, self.cnt_timestep)
        
        known_victims_pos = [r['pos'] for r in self.victim_manager.registry]
        nearby_drones_pos = []
        
        check_center = False
        
        # Parse semantic sensor data
        if semantic_data:
            min_dist_to_center = float('inf')
            for data in semantic_data:
                angle_global = self.estimated_angle + data.angle
                obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                
                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    nearby_drones_pos.append(np.array([obj_x, obj_y]))
                    
                elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    check_center = True
                    dist_to_center = np.linalg.norm(self.estimated_pos - np.array([obj_x, obj_y]))
                    if dist_to_center < min_dist_to_center:
                        min_dist_to_center = dist_to_center
                        self.rescue_center_pos = np.array([obj_x, obj_y])

        # Update SLAM
        self.nav.update_navigator(nearby_drones=nearby_drones_pos, nearby_victims=known_victims_pos)

        # Record Initial Position
        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()
            print(f"[{self.identifier}] 🏁 MISSION STARTED.")

        # ================= 2. COMMUNICATION =================
        self.comms.process_incoming_messages()
        # print(f"List victim taken care of: {self.comms.list_victims_taken_care_of}")

        # Debug visualization
        if self.cnt_timestep % 5 == 0:
            self.nav.obstacle_map.display(
                self.estimated_pos, 
                current_target=self.current_target,
                current_path=self.nav.current_path, 
                window_name=f"Map - Drone {self.identifier}"
            )

        # ================= 3. STATE MACHINE =================

        # --- STATE: DISPERSING ---
        # Objective: Spread out from the starting cluster to cover maximum area quickly.
        if self.state == "DISPERSING":
            dx = self.estimated_pos[0] - self.initial_position[0]
            dy = self.estimated_pos[1] - self.initial_position[1]
            dist_moved = math.hypot(dx, dy)
            
            # Check escape conditions: Did we move far enough from the cluster, or did we timeout?
            if self.safe_dispersion_reached_tick is None:
                if self.cnt_timestep >= 100:
                    self.safe_dispersion_reached_tick = self.cnt_timestep
            
            # Phase 1: Active Repulsion (Push away from peers using Pilot's repulsive force)
            if self.safe_dispersion_reached_tick is None:
                return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=0, repulsive_force_bool=True)
            
            # Phase 2: Scanning Mode (Spin 360 degrees to map local surroundings)
            elif self.cnt_timestep < self.safe_dispersion_reached_tick + 50: 
                return self.pilot.move_function(forward=0, lateral=0, rotation=1, grasper=0, repulsive_force_bool=True)
            
            # Phase 3: Smart Fan Dispersion (Calculate optimal spread angle based on map boundaries)
            else:
                W, H = self.map_size
                sx, sy = self.initial_position
                
                # 1. Measure distance from spawn point to the 4 map boundaries
                # (Assuming map center is (0,0), bounds are -W/2 to W/2 and -H/2 to H/2)
                dist_L = sx - (-W / 2.0)
                dist_R = (W / 2.0) - sx
                dist_T = sy - (-H / 2.0)
                dist_B = (H / 2.0) - sy
                
                vx, vy = 0.0, 0.0
                MARGIN = 200.0 # Distance threshold (px) to consider a wall "close"
                
                # 2. Create repulsion vector - Face away from the closest walls
                if dist_L < MARGIN: vx += 1.0
                if dist_R < MARGIN: vx -= 1.0
                if dist_T < MARGIN: vy += 1.0
                if dist_B < MARGIN: vy -= 1.0
                
                TOTAL_DRONES = 10 
                safe_id = self.identifier % TOTAL_DRONES # Prevent out-of-bounds just in case
                
                # 3. Calculate Fan Spread and assign individual angles
                if vx == 0.0 and vy == 0.0:
                    # Case 1: Spawned in the center of the map -> 360-degree circular spread
                    self.preferred_angle = (safe_id / TOTAL_DRONES) * 2 * math.pi
                else:
                    # Case 2: Spawned near an edge or corner -> Directed fan spread
                    base_angle = math.atan2(vy, vx)
                    walls_touching = abs(vx) + abs(vy)
                    
                    if walls_touching >= 2.0:
                        # Corner spawn (e.g., Bottom-Left): 90-degree physical corner -> 80-degree fan (0.44 pi)
                        spread = math.pi * 0.44 
                    else:
                        # Edge spawn (e.g., Left wall): 180-degree physical wall -> 140-degree fan (0.77 pi)
                        spread = math.pi * 0.77 
                        
                    start_angle = base_angle - (spread / 2.0)
                    
                    # Distribute angles evenly among the 10 drones
                    self.preferred_angle = start_angle + (safe_id / max(1, (TOTAL_DRONES - 1))) * spread
                    
                # Normalize angle to [-pi, pi] to avoid math errors in scoring algorithms
                self.preferred_angle = math.atan2(math.sin(self.preferred_angle), math.cos(self.preferred_angle))

                print(f"[{self.identifier}] 🚀 SMART DISPERSION. Target Angle: {math.degrees(self.preferred_angle):.0f}°. To EXPLORING.")
                self.state = "EXPLORING"

        # --- SYSTEM CHECKS: BATTERY & PANIC MODE ---
        self.steps_remaining = self.max_timesteps - self.cnt_timestep
        self.pilot.low_battery()

        # --- ANTI-STUCK & PANIC MODE ---
        is_currently_stuck = self.nav.is_stuck()
            
        if self.state != "END_GAME":
            # 1. Activate Panic Mode if drone is actively stuck
            if is_currently_stuck:
                self.panic_timer = 80 
                
        # 2. Synchronize Panic Mode with Cost Map (Smart Hysteresis)
        if self.panic_timer > 0:
            self.panic_timer -= 1
            
            # Turn ON Panic Mode (Flatten cost map to squeeze through)
            if not getattr(self.nav.obstacle_map, 'panic_mode', False):
                self.nav.obstacle_map.panic_mode = True
                self.nav.obstacle_map.update_cost_map() 
                print(f"[{self.identifier}] 🚨 PANIC MODE ON: Flattening Cost Map to escape narrow corridor!")
                
        else:
            # 3. [NEW] SAFE EXIT CHECK
            # Before turning Panic Mode OFF, verify the drone is in a wide, open space.
            # Turning it off while still in a narrow gap will instantly inflate the lethal cost 
            # and freeze the drone again (Oscillation bug).
            if getattr(self.nav.obstacle_map, 'panic_mode', False):
                safe_to_relax = True
                
                # Check distance to nearest wall using the existing dist_map
                if hasattr(self.nav.obstacle_map, 'dist_map') and self.nav.obstacle_map.dist_map is not None:
                    gx, gy = self.nav.obstacle_map.world_to_grid(self.estimated_pos[0], self.estimated_pos[1])
                    if 0 <= gx < self.nav.obstacle_map.grid_w and 0 <= gy < self.nav.obstacle_map.grid_h:
                        dist_to_wall = self.nav.obstacle_map.dist_map[gy, gx]
                        # If distance to wall is less than 2.0 grids (~16cm), delay turning off
                        if dist_to_wall < 2.0:
                            safe_to_relax = False
                            self.panic_timer = 20 # Add bonus time to keep escaping
                
                # Turn OFF Panic Mode ONLY if physically safe
                if safe_to_relax:
                    self.nav.obstacle_map.panic_mode = False
                    self.nav.obstacle_map.update_cost_map() 
                    print(f"[{self.identifier}] 😌 PANIC MODE OFF: Wide space reached, normal navigation resumed.")

        # 4. Emergency wiggle maneuver
        if is_currently_stuck:
            self.nav.current_path = []
            grasper_state = 1 if self.grasped_wounded_persons() else 0
            return self.pilot.move_function(forward=0, lateral=0, rotation=1.0, grasper=grasper_state, repulsive_force_bool=True)

        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0: 
                self.blacklist_timer -= 1
            else: 
                self.blacklisted_targets = []

            # Priority 1: Check for known victims
            best_victim_pos = self.victim_manager.get_nearest_victim(self.estimated_pos, self.blacklisted_targets)
            self.current_target_best_victim_pos = best_victim_pos
            if best_victim_pos is not None:
                self.current_target = best_victim_pos
                self.state = "RESCUING"
                self.path_fail_count = 0 
                print(f"[{self.identifier}] 🚑 RESCUING VICTIM at {best_victim_pos}")

            # Priority 2: Find Frontier via Floodfill
            if self.state == "EXPLORING": 
                if self.floodfill_cooldown > 0: self.floodfill_cooldown -= 1
                
                if self.current_target is None and self.floodfill_cooldown <= 0:
                    frontier, path = self.nav.obstacle_map.get_reachable_frontier_and_path(
                        self.estimated_pos, self.estimated_angle, self.preferred_angle,
                        self.initial_position, self.rescue_center_pos,
                        self.blacklisted_targets
                    )
                    
                    if frontier is not None:
                        self.current_target = frontier
                        self.nav.current_path = path 
                        self.nav.last_path_index = 0
                        self.nav.last_astar_target = frontier.copy() 
                        self.path_fail_count = 0
                    else:
                        self.current_target = None
                        self.floodfill_cooldown = 15
                    
                    # Fallback: Lost in the fog. Spin to clear Lidar data.
                    if self.current_target is None:
                        return self.pilot.move_function(forward=0, lateral=0, rotation=0.8, grasper=0, repulsive_force_bool=True)

        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":
            # --- [NEW] ANTI-STEAL ABORT (Stop if victim is rescued by other drone) ---
            if self.current_target is not None:
                is_taken = self.victim_manager.is_victim_taken_care_of(self.current_target)
                
                is_still_valid = False
                for record in self.victim_manager.registry:
                    if np.linalg.norm(record['pos'] - self.current_target) < 20.0:
                        is_still_valid = True; break
                
                if is_taken or not is_still_valid:
                    print(f"[{self.identifier}] 🛑 Target rescued by teammate. Aborting rescue!")
                    self.current_target = None
                    self.state = 'EXPLORING'
                    self.nav.current_path = []
                    self.rescue_time = 0
                    return self.pilot.move_function(forward=0, lateral=0, rotation=1, grasper=0, repulsive_force_bool=True)
            # -------------------------------------------------------------
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target) if self.current_target is not None else 9999
            if dist_to_target < 70: 
                self.rescue_time += 1

            # Handle Ghost Victims / Timeouts
            if self.rescue_time >= 50:
                if self.current_target is not None: 
                    self.victim_manager.delete_victim_at(self.current_target)
                self.current_target = None
                self.state = 'EXPLORING'
                self.nav.current_path = []
                self.rescue_time = 0
                print(f"[{self.identifier}] Change to EXPLORING!")
                return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=0, repulsive_force_bool=True)
            
            # Successful Grasp
            if self.grasped_wounded_persons():
                self.rescue_time = 0
                self.state = "RETURNING"
                self.victim_manager.delete_victim_at(self.estimated_pos)
                self.current_target = self.initial_position if self.initial_position is not None else self.rescue_center_pos
                print(f"[{self.identifier}] ✅ VICTIM SECURED. Returning to Base.")

        # --- STATE: RETURNING & DROPPING ---
        elif self.state == "RETURNING":
            if self.current_target is None: 
                self.current_target = self.initial_position
            if np.linalg.norm(self.estimated_pos - self.current_target) < 60.0 and check_center:
                self.state = "DROPPING"

        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            if check_center: 
                self.drop_step += 1
            
            # Drop completion condition
            if self.drop_step > 150 or not self.grasped_wounded_persons(): 
                self.drop_step = 0
                self.nav.current_path = []
                self.current_target = None
                print(f"[{self.identifier}] ⏬ VICTIM DROPPED. Resuming Exploration.")
                self.state = "EXPLORING" 
                return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=0, repulsive_force_bool=True)              
            return self.pilot.move_to_target_carrot()

        # --- STATE: END GAME ---
        elif self.state == "END_GAME":
            if not self.is_inside_return_area:
                self.state = "RETURNING"
            return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=0, repulsive_force_bool=True)


        # ================= 4. EXECUTION & TARGET VALIDATION =================
        next_waypoint = None

        if self.current_target is not None:
            dist = np.linalg.norm(self.estimated_pos - self.current_target)
            
            # Validate pathfinding health
            if len(self.nav.current_path) == 0 and dist > 40.0:
                self.path_fail_count += 1 
                if self.path_fail_count > 30:
                    print(f"[{self.identifier}] ❌ PATHFINDING FAILED. Blacklisting target.")
                    if self.state != "RETURNING":
                        self.blacklisted_targets.append(self.current_target)
                    
                    self.blacklist_timer = 200
                    self.current_target = None
                    self.path_fail_count = 0
                    
                    grasper_state = 1 if self.grasped_wounded_persons() else 0
                    return self.pilot.move_function(forward=0, lateral=0, rotation=1.0, grasper=grasper_state, repulsive_force_bool=True)
            else:
                self.path_fail_count = 0
            
            # Request route from Navigator
            next_waypoint = self.nav.get_next_waypoint(self.current_target)

        # --- Robust Arrival Check for Exploring ---
        if self.state == "EXPLORING" and self.current_target is not None:
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)
            arrived_close = (dist_to_target < 35.0)
            
            gx, gy = self.nav.obstacle_map.world_to_grid(self.current_target[0], self.current_target[1])
            cell_value = self.nav.obstacle_map.grid[gy, gx] if 0 <= gx < self.nav.obstacle_map.grid_w and 0 <= gy < self.nav.obstacle_map.grid_h else 0.0
            
            target_obstructed = (cell_value > 10.0)
            
            # Check for Stale Target (Area already explored by teammates)
            target_stale = True
            search_radius = 2 # 5x5 window
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= nx < self.nav.obstacle_map.grid_w and 0 <= ny < self.nav.obstacle_map.grid_h:
                        if -0.1 < self.nav.obstacle_map.grid[ny, nx] < 5.0:
                            target_stale = False
                            break
                if not target_stale: break

            if arrived_close or target_obstructed or target_stale:
                self.current_target = None
                self.nav.current_path = []
                grasper_state = 1 if self.grasped_wounded_persons() else 0
                return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=grasper_state, repulsive_force_bool=True)

        # --- Handle Route Blockages during Active Mission ---
        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
             if next_waypoint is None:
                self.patience += 1
                if self.patience > 50:
                    self.patience = 0
                    if self.state == "RETURNING" and self.initial_position is not None:
                         if np.linalg.norm(self.current_target - self.initial_position) > 10.0:
                             self.current_target = self.initial_position
                             return self.pilot.move_function(forward=0, lateral=0, rotation=0, grasper=1, repulsive_force_bool=True)
                    
                    grasped = 1 if self.grasped_wounded_persons() else 0
                    return self.pilot.move_function(forward=0, lateral=1, rotation=0.5, grasper=grasped, repulsive_force_bool=True)

        # --- Fast Unstuck Check ---
        if next_waypoint is None:
            grasper_state = 1 if self.grasped_wounded_persons() else 0
            if self.state == "EXPLORING" and self.current_target is not None:
                self.floodfill_cooldown = 15
                self.current_target = None
                self.nav.current_path = []
            
            return self.pilot.move_function(forward=0, lateral=0, rotation=1, grasper=grasper_state, repulsive_force_bool=True)
        
        # ================= 5. FINAL COMMAND DISPATCH =================
        real_target = self.current_target 
        self.current_target = next_waypoint 
        self.MAX_SPEED = 1.0 if self.state == "RETURNING" else 0.9
        
        command = self.pilot.move_to_target_carrot()
        
        self.current_target = real_target 
        return command

    def define_message_for_all(self):
        return self.comms.create_new_message()
    
    def draw_top_layer(self):
        self.draw_identifier()