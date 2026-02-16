import math
import random 
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

try:
    from .navigator import Navigator
    from .pilot import Pilot
    from .victim_manager import VictimManager
except ImportError:
    from navigator import Navigator
    from pilot import Pilot
    from victim_manager import VictimManager

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
        self.victim_manager = VictimManager()
        
        # --- STATE VARIABLES ---
        self.state = "DISPERSING" 
        self.current_target = None 
        self.rescue_center_pos = None 
        self.initial_position = None 
        self.cnt_timestep = 0
        
        self.last_rescue_pos = None
        self.drop_step = 0

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

    def control(self) -> CommandsDict:
        """
        Main control loop called by the simulator every step.
        """
        self.cnt_timestep += 1
        
        # 1. Update Position & Neighbors
        nearby_drones_pos = []
        semantic_data = self.semantic_values()
        if semantic_data:
            for data in semantic_data:
                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    angle_global = self.estimated_angle + data.angle
                    dx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                    dy = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                    nearby_drones_pos.append(np.array([dx, dy]))

        self.nav.update_navigator(nearby_drones=nearby_drones_pos)
        
        # 2. Update Victim Registry
        self.victim_manager.update_from_sensor(
            self.estimated_pos,
            self.estimated_angle,
            semantic_data,
            self.cnt_timestep
        )

        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()
            print(f"[{self.identifier}] 🏁 STARTED at {self.initial_position}")

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
            print(f"[{self.identifier}] 🚫 Masked Rescue Center area.")

        # Debug visualization (Drone 0 only)
        if self.cnt_timestep % 5 == 0 and self.identifier == 0:
            self.nav.obstacle_map.display(
                self.estimated_pos, 
                current_target=self.current_target,
                current_path=self.nav.current_astar_path, 
                window_name=f"Obstacle Map - Drone {self.identifier}"
            )

        # ================= STATE MACHINE =================

        # --- STATE: DISPERSING ---
        # Initial spread to avoid collisions at spawn.
        if self.state == "DISPERSING":
            if self.cnt_timestep < 20: 
                _, lat_drone = self.pilot.calculate_repulsive_force()
                return {"forward": 0.0, "lateral": np.clip(lat_drone, -1, 1), "rotation": 0.0, "grasper": 0}
            else:
                print(f"[{self.identifier}] 🚀 EXPLORING!")
                self.state = "EXPLORING"

        # --- GLOBAL: Battery Check ---
        steps_remaining = self.max_timesteps - self.cnt_timestep
        RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.1) # Return when 10% battery remains
        if steps_remaining <= RETURN_TRIGGER_STEPS:
            if self.is_inside_return_area: self.state = "END_GAME"
            else:
                if self.state not in ["RETURNING", "DROPPING", "END_GAME"]:
                    print(f"[{self.identifier}] 🔋 LOW BATTERY! Returning home.")
                    self.state = "RETURNING"
                    self.current_target = None 

        # --- GLOBAL: Anti-Stuck Logic ---
        # Checks if position hasn't changed significantly over a time window.
        self.pos_history_long.append(self.estimated_pos.copy())
        waiting = 130 if self.state == 'EXPLORING' else 150
        if len(self.pos_history_long) > waiting: self.pos_history_long.pop(0) 
        if self.state not in ["END_GAME", "DISPERSING"] and len(self.pos_history_long) == waiting and steps_remaining > RETURN_TRIGGER_STEPS:
            start_pos = self.pos_history_long[0]
            dist_moved = np.linalg.norm(self.estimated_pos - start_pos)
            if dist_moved < 8.0:
                print(f"[{self.identifier}] ⚠️ STUCK DETECTED! Replanning...")
                self.nav.current_astar_path = []
                # Random maneuver to break free
                fwd = 0; lat = 1.0 if random.random() > 0.5 else -1.0
                grasper = 1 if self.grasped_wounded_persons() else 0
                return {"forward": fwd, "lateral": lat, "rotation": 0.0, "grasper": grasper}

        # --- STATE: EXPLORING ---
        if self.state == "EXPLORING":
            if self.blacklist_timer > 0: self.blacklist_timer -= 1
            else: self.blacklisted_targets = []

            # 1. Check for Victims (Highest Priority)
            best_victim_pos = self.victim_manager.get_nearest_victim(self.estimated_pos)

            # Ignore victims at home base (already rescued)
            if best_victim_pos is not None and self.rescue_center_pos is not None:
                dist_to_home = np.linalg.norm(best_victim_pos - self.rescue_center_pos)
                if dist_to_home < 100.0:
                    self.victim_manager.delete_victim_at(best_victim_pos)
                    best_victim_pos = None 

            if best_victim_pos is not None:
                self.current_target = best_victim_pos
                self.state = "RESCUING"
                self.path_fail_count = 0 
                print(f"[{self.identifier}] 🚑 FOUND VICTIM at {best_victim_pos}")

            # 2. Find Frontier (Exploration)
            if self.state == "EXPLORING": 
                if self.current_target is None:
                    frontier = self.nav.obstacle_map.get_frontier_target(
                        self.estimated_pos, 
                        self.estimated_angle,
                        drone_id=self.identifier,        
                        current_step=self.cnt_timestep,
                        busy_targets=[], 
                        nearby_drones=nearby_drones_pos,
                        rescue_center_pos=self.rescue_center_pos
                    )
                    
                    # Validate Frontier against blacklist
                    if frontier is not None:
                        is_bad = False
                        for bad in self.blacklisted_targets:
                            if np.linalg.norm(frontier - bad) < 20.0: is_bad = True; break
                        
                        if not is_bad:
                            self.current_target = frontier
                            self.path_fail_count = 0 
                        else: frontier = None 

                    # Fallback 1: Random unknown point
                    if self.current_target is None:
                        unknown_target = self.nav.obstacle_map.get_unknown_target(
                            self.estimated_pos, nearby_drones=nearby_drones_pos
                        )
                        if unknown_target is not None:
                            self.current_target = unknown_target
                            self.path_fail_count = 0

                    # Fallback 2: Random known free point (just to keep moving)
                    if self.current_target is None:
                        random_free = self.nav.obstacle_map.get_random_free_target(self.estimated_pos)
                        if random_free is not None: 
                            self.current_target = random_free
                            self.path_fail_count = 0
                        else: return {"forward": 0.0, "lateral": 0.0, "rotation": 0.8, "grasper": 0}

        # --- STATE: RESCUING ---
        elif self.state == "RESCUING":
            dist_to_target = 9999
            if self.current_target is not None:
                dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)
            if dist_to_target < 40: self.rescue_time += 1

            # Logic: Ray Walking to find a safe standing point near the victim
            victim_in_sight = False
            if semantic_data:
                for data in semantic_data:
                    if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                        victim_in_sight = True
                        angle_global = self.estimated_angle + data.angle
                        best_safe_dist = 0.0 
                        check_range = np.arange(max(0.0, data.distance - 20.0), 0, -10.0)
                        found_valid_spot = False
                        # Walk backwards from victim to find safe spot (Cost < 300)
                        for d in check_range:
                            cx = self.estimated_pos[0] + d * math.cos(angle_global)
                            cy = self.estimated_pos[1] + d * math.sin(angle_global)
                            if self.nav.obstacle_map.get_cost_at(np.array([cx, cy])) < 300.0:
                                best_safe_dist = d; found_valid_spot = True; break
                        if not found_valid_spot: best_safe_dist = 20.0 

                        safe_vx = self.estimated_pos[0] + best_safe_dist * math.cos(angle_global)
                        safe_vy = self.estimated_pos[1] + best_safe_dist * math.sin(angle_global)
                        self.current_target = np.array([safe_vx, safe_vy])
                        break 

            # Rescue Timeout
            if self.rescue_time >= 150:
                if self.current_target is not None:
                     self.victim_manager.delete_victim_at(self.current_target)
                     print(f'[{self.identifier}] Delete current victim target!')
                self.current_target = None
                self.state = 'EXPLORING'; self.nav.current_astar_path = []; self.rescue_time = 0
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            
            # Successful Grasp
            if self.grasped_wounded_persons():
                self.rescue_time = 0; self.state = "RETURNING"
                self.victim_manager.delete_victim_at(self.estimated_pos)
                self.current_target = self.initial_position
                if self.current_target is None and self.rescue_center_pos is not None:
                    self.current_target = self.rescue_center_pos
                print(f"[{self.identifier}] ✅ GRASPED! Returning home.")

        # --- STATE: RETURNING ---
        elif self.state == "RETURNING":
            if self.current_target is None:
                self.current_target = self.initial_position if self.initial_position is not None else self.rescue_center_pos
            if np.linalg.norm(self.estimated_pos - self.current_target) < 50.0 and steps_remaining > RETURN_TRIGGER_STEPS:
                self.state = "DROPPING"

        # --- STATE: DROPPING ---
        elif self.state == "DROPPING":
            self.current_target = self.rescue_center_pos
            if check_center: self.drop_step += 1
            # Wait for drop confirmation or timeout
            if self.drop_step > 150 or not self.grasped_wounded_persons(): 
                print(f"[{self.identifier}] ⏬ DROPPED! Going Explore.")
                self.drop_step = 0
                self.nav.current_astar_path = []
                self.state = "EXPLORING" 
                self.current_target = None
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return self.pilot.move_to_target_carrot()

        # --- STATE: END GAME ---
        elif self.state == "END_GAME":
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # ================= EXECUTION =================
        next_waypoint = None

        # [PATH VALIDATION] Check if Navigator failed to find a path
        if self.current_target is not None:
            dist = np.linalg.norm(self.estimated_pos - self.current_target)
            
            # If path is empty but we are far from target, A* failed.
            if len(self.nav.current_astar_path) == 0 and dist > 40.0:
                self.path_fail_count += 1 
                
                # Patience Check: Only blacklist if it fails consistently for > 30 ticks
                if self.path_fail_count > 30:
                    print(f"[{self.identifier}] ❌ Path failed to {self.current_target} ({self.path_fail_count} ticks). Blacklisting!")
                    self.blacklisted_targets.append(self.current_target)
                    self.blacklist_timer = 200 
                    self.current_target = None 
                    self.path_fail_count = 0
                    return {"forward": 0.0, "lateral": 0.0, "rotation": 1.0, "grasper": 0} 
                else:
                    pass # Give it more time to replan
            else:
                self.path_fail_count = 0

        if self.current_target is not None:
            next_waypoint = self.nav.get_next_waypoint(self.current_target)
        
        # Fallback Logic if path is empty (Wiggle to unstick)
        dist_to_target = 9999.0
        if self.current_target is not None:
            dist_to_target = np.linalg.norm(self.estimated_pos - self.current_target)

        path_is_empty = (len(self.nav.current_astar_path) == 0)

        if self.state == "EXPLORING" and self.current_target is not None:
            if dist_to_target < 40.0:
                self.current_target = None
                self.nav.current_astar_path = []
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0, "grasper": 0}

        elif self.state in ["RETURNING", "RESCUING"] and self.current_target is not None:
            if path_is_empty and dist_to_target > 40.0:
                self.patience += 1
                if self.patience > 50:
                    self.patience = 0
                    # If stuck returning, reset target to initial pos
                    if self.state == "RETURNING" and self.initial_position is not None:
                         if np.linalg.norm(self.current_target - self.initial_position) > 10.0:
                             self.current_target = self.initial_position
                             return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
                    return {"forward": 0.0, "lateral": 1.0, "rotation": 0.5, "grasper": 1 if self.grasped_wounded_persons() else 0}

        if next_waypoint is None:
             return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1 if self.grasped_wounded_persons() else 0}
        
        # Execute Pilot Command
        real_target = self.current_target 
        self.current_target = next_waypoint 
        
        command = self.pilot.move_to_target_carrot()
        self.current_target = real_target 
        return command

    def define_message_for_all(self):
        pass