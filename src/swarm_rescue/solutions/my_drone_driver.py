import math
import random # [MỚI] Import random để random hướng giãy
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

# IMPORT COMPONENTS
#try:
from .navigator import Navigator
from .pilot import Pilot
from .communicator import CommunicatorHandler
#except ImportError:
#    from navigator import Navigator
#    from pilot import Pilot
#    from communicator import CommunicatorHandler

class MyStatefulDrone(DroneAbstract):
    
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        
        # --- INITIALIZE COMPONENTS ---
        self.nav = Navigator(self)
        self.pilot = Pilot(self)
        self.comms = CommunicatorHandler(self)
        
        # --- STATE VARIABLES ---
        self.state = "EXPLORING"
        self.current_target = None 
        self.rescue_center_pos = None 
        self.position_before_rescue = None
        self.initial_position = None
        self.cnt_timestep = 0
        self.same_position_timestep = 0
        self.previous_position = None
        
        self.last_rescue_pos = None
        self.initial_spot_pos = None
        self.found_person_pos = None
        self.patience = None
        self.not_grapsed = False
        self.drop_step = 0
        self.priority = self.comms.avoidance_priority()

        self.estimated_pos = np.array([0.0, 0.0]) 
        self.estimated_angle = 0.0
        self.gps_last_known = None 
        
        # Save history position for detect stuck
        self.pos_history_long = []

        # Config
        misc_data = kwargs.get('misc_data')
        self.max_timesteps = 2700 
        self.map_size = (800, 600) 
        if misc_data:
            self.max_timesteps = misc_data.max_timestep_limit
            self.map_size = misc_data.size_area

    def control(self) -> CommandsDict:
        self.cnt_timestep += 1
        check_center = False
        
        # 1. Update Navigator & Sensors
        self.nav.update_navigator()
        REACH_THRESHOLD_LOCAL = 25 if self.grasped_wounded_persons() else 40
        
        if self.cnt_timestep == 1:
            self.initial_position = self.estimated_pos.copy()

        # 2. Semantic Sensor Scan
        semantic_data = self.semantic_values()
        closest_person_dist = float('inf')
        closest_person_pos = None
        
        if semantic_data:
            for data in semantic_data:
                angle_global = self.estimated_angle + data.angle
                obj_x = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                obj_y = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    if data.distance < closest_person_dist:
                        closest_person_dist = data.distance
                        closest_person_pos = np.array([obj_x, obj_y])
                
                elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    check_center = True
                    if self.rescue_center_pos is None:
                       self.rescue_center_pos = np.array([obj_x, obj_y])
            
            if closest_person_pos is not None:
                if not self.comms.is_target_taken_or_better_candidate(closest_person_pos):
                    ALPHA = 0.7
                    if self.found_person_pos is None:
                        self.found_person_pos = closest_person_pos
                    else:
                        self.found_person_pos = ALPHA * closest_person_pos + (1 - ALPHA) * self.found_person_pos

        # 3. Force return when remaining timestep is low to check health
        steps_remaining = self.max_timesteps - self.cnt_timestep
        RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        # FORCE RETURN
        if steps_remaining <= RETURN_TRIGGER_STEPS:
            if self.is_inside_return_area: self.state = "END_GAME"
            else:
                if steps_remaining == RETURN_TRIGGER_STEPS: print('Time to return to check health!')
                if self.state != "DROPPING":
                    if self.state == "COMMUTING":
                        self.current_target = None 

                    if self.current_target is None:
                        min_dist = float('inf')
                        nearest_node = None
                        if self.nav.path_history:
                            for key in self.nav.path_history.keys():
                                node_pos = np.array(key)
                                d = np.linalg.norm(self.estimated_pos - node_pos)
                                if d < min_dist:
                                    min_dist = d
                                    nearest_node = node_pos
                        
                        if nearest_node is not None:
                            self.current_target = nearest_node
                        else:
                            if self.rescue_center_pos is not None:
                                self.current_target = self.rescue_center_pos
                            else:
                                self.current_target = self.initial_position
                    self.state = "RETURNING"
                    self.not_grapsed = True
                    print(f"Switched to RETURNING")

        # =========================================================================
        # 5. [NEW] HARD STUCK DETECTION & UNSTICK
        # =========================================================================
        
        ## Update history of current position (Sliding Window 60 steps)
        #self.pos_history_long.append(self.estimated_pos.copy())
        #if len(self.pos_history_long) > 60:
        #    self.pos_history_long.pop(0) 
        #
        ## Check stuck (when the list has 60 elements)
        #if len(self.pos_history_long) == 60 and (steps_remaining > RETURN_TRIGGER_STEPS or (steps_remaining <= RETURN_TRIGGER_STEPS and not self.is_inside_return_area)):
        #    # Compare current position with 60 steps before position
        #    start_pos = self.pos_history_long[0]
        #    dist_moved = np.linalg.norm(self.estimated_pos - start_pos)
        #    
        #    # If move only < 17px in 60 steps -> Stuck
        #    if dist_moved < 17.0:
        #        # 3. Action to unstuck
        #        
        #        lat_force = 0.0
        #        
        #        # Check target direction to slide (if right -> slide right, if left -> slide left)
        #        if self.current_target is not None:
        #            d_x = self.current_target[0] - self.estimated_pos[0]
        #            d_y = self.current_target[1] - self.estimated_pos[1]
        #            target_angle = math.atan2(d_y, d_x)
        #            
        #            # Calculate dif angle to the front of the drone
        #            angle_diff = target_angle - self.estimated_angle
        #            
        #            # Normalize to range [-pi, pi]
        #            while angle_diff > math.pi: angle_diff -= 2 * math.pi
        #            while angle_diff <= -math.pi: angle_diff += 2 * math.pi
        #            
        #            # If target on the left (> 0) -> Slide left (1.0)
        #            # If target on the right (< 0) -> Slide right (-1.0)
        #            lat_force = 1.0 if angle_diff > 0 else -1.0
        #        else:
        #            # Fallback if there was no target (random)
        #            lat_force = 1.0 if random.random() > 0.5 else -1.0
#
        #        # Go back to unstuck with lateral movement for better aim to target
        #        fwd_force = -0.7
        #        
        #        # Keep the grasping state
        #        grasper_val = 1 if self.grasped_wounded_persons() else 0
        #        
        #        # Return movement command
        #        return {
        #            "forward": fwd_force, 
        #            "lateral": lat_force,
        #            "rotation": 0.0,      # No rotation for constant direction of drone
        #            "grasper": grasper_val
        #        }
            
        # =========================================================================
        # 5. [NEW][NEW] HARD STUCK DETECTION & UNSTICK (V2)
        # =========================================================================


         

        if self.current_target is not None:

            print(self.identifier, self.estimated_pos, self.current_target)

            if self.previous_position is None:
                
                self.previous_position = self.current_target

                if np.hypot(self.previous_position[0] - self.current_target[0],self.previous_position[1] - self.current_target[1]) < 25:

                    self.same_position_timestep += 1

                else:

                    self.previous_position = self.current_target
                    self.same_position_timestep = 0

        # ================= STATE MACHINE =================

        # --- EXPLORING ---
        if self.state == "EXPLORING":
            # print('exploring')
            if self.found_person_pos is not None:
                if self.comms.is_target_taken_or_better_candidate(self.found_person_pos):
                    self.nav.visit(self.found_person_pos) 
                    self.found_person_pos = None 
                else:
                    self.state = "RESCUING"
                    print(f"Switched to RESCUING")
                    self.position_before_rescue = self.current_target 
                    if self.position_before_rescue is None: self.position_before_rescue = self.estimated_pos
                    self.current_target = self.found_person_pos
                    self.initial_spot_pos = self.found_person_pos.copy()
            
            elif self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                if self.current_target is not None: self.nav.visit(self.current_target)
                else: self.nav.visit(self.estimated_pos)
                self.nav.update_mapper()
                # print('updated mapper')                
                pos_key = (int(self.estimated_pos[0]), int(self.estimated_pos[1]))
                next_target = None
                
                # Smart target choosing, choose target far from rescue center (no person near rescue center and not worth exploring)
                if pos_key in self.nav.edge and len(self.nav.edge[pos_key]) > 0:
                    while len(self.nav.edge[pos_key]) > 0:
                        candidate = self.nav.edge[pos_key].pop() 
                        is_safe = True
                        if self.rescue_center_pos is not None:
                            dist_to_rc = np.linalg.norm(np.array(candidate) - self.rescue_center_pos)
                            if dist_to_rc < 150.0: is_safe = False
                        
                        if is_safe:
                            next_target = candidate
                            break
                if next_target is not None:
                    target_int_key = (int(next_target[0]), int(next_target[1]))
                    if self.current_target is None: self.nav.path_history[target_int_key] = self.estimated_pos.copy()
                    else: self.nav.path_history[target_int_key] = self.current_target.copy()
                    self.current_target = np.array(next_target)
                    #print(f"Explore location {self.current_target}")
                else:
                    if self.current_target is not None:
                        current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                        if current_int_key in self.nav.path_history:
                             self.current_target = self.nav.path_history[current_int_key]
                    else:
                        self.current_target = self.estimated_pos.copy()
                    print(f"Goes back to {self.current_target}")

            elif np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                pass
                # if self.nav.is_path_blocked(self.current_target):
                #     print(f"Current target is blocked by wall!")
                #     if self.pilot.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5): pass 
                #     else:
                #         bypass_node = self.nav.find_best_bypass(self.current_target)
                #         if bypass_node is not None:
                #             target_key = (int(self.current_target[0]), int(self.current_target[1]))
                #             if target_key in self.nav.path_history:
                #                 parent_of_blocked = self.nav.path_history[target_key]
                #                 bypass_key = (int(bypass_node[0]), int(bypass_node[1]))
                #                 self.nav.path_history[bypass_key] = parent_of_blocked
                #                 self.current_target = bypass_node
                #             else:
                #                 self.current_target = bypass_node

        # --- RESCUING ---
        elif self.state == "RESCUING":
            if self.current_target is not None:
                if self.comms.is_target_taken_or_better_candidate(self.current_target):
                    self.state = "EXPLORING"
                    print(f"Switched to EXPLORING because of other drone")
                    self.found_person_pos = None
                    self.current_target = self.position_before_rescue
                    if self.current_target is None:
                        return {"forward": -0.5, "lateral": 0.5, "rotation": 0.0, "grasper": 0}
                    return self.pilot.move_to_target_PID()

            if not self.grasped_wounded_persons():
                if self.patience is None: self.patience = 0
                self.patience += 1
                if self.patience > 80:
                    self.found_person_pos = None
                    self.patience = None
                    self.state = "EXPLORING"
                    print(f"Switched to EXPLORING because victim is rescued by other drones, goes back to {self.position_before_rescue}")
                    self.current_target = self.position_before_rescue
                    return self.pilot.move_to_target_PID()

            if self.grasped_wounded_persons():
                self.last_rescue_pos = self.current_target.copy()
                self.state = "RETURNING"
                print(f"Switched to RETURNING, goes back to {self.position_before_rescue}")
                self.current_target = self.position_before_rescue
                if self.current_target is None:
                    self.current_target = self.rescue_center_pos

        # --- RETURNING ---
        elif self.state == "RETURNING":
            if check_center and self.rescue_center_pos is not None:
                print(f'See rescue center!')
                if not self.grasped_wounded_persons(): 
                    self.state = "DROPPING"
                    print(f"Switched to DROPPING")
                self.current_target = self.rescue_center_pos

                if np.linalg.norm(self.estimated_pos - self.current_target) < 10.0:
                    print(f"Switched to DROPPING")
                    self.state = "DROPPING"
                    self.drop_step = 0
            else:
                # print(f'Going back to {self.current_target}, dist is {np.linalg.norm(self.estimated_pos - self.current_target)}!')
                if self.current_target is None: 
                    self.current_target = self.position_before_rescue
                
                if self.current_target is None:
                    if self.rescue_center_pos is not None: self.current_target = self.rescue_center_pos
                    else: self.current_target = self.initial_position

                # if self.current_target is not None and self.cnt_timestep % 5 == 0:
                #     if np.linalg.norm(self.estimated_pos - self.current_target) > REACH_THRESHOLD_LOCAL:
                #         shortcut = self.nav.find_shortcut_target()
                #         if shortcut is not None: 
                #             print(f'Found shortcut to {shortcut}')
                #             self.current_target = shortcut
                
                if self.current_target is not None and np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                    if np.linalg.norm(self.current_target - self.initial_position) < 10:
                        self.state = 'END_GAME'
                        print(f"Switched to END_GAME")
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
                    
                    current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                    if current_int_key in self.nav.path_history: 
                        parent_node = self.nav.path_history[current_int_key]
                        self.nav.waypoint_stack.append(self.current_target.copy())
                        self.current_target = parent_node 
                    else:
                        print(f"Doesn't see key in history!")
                        if self.rescue_center_pos is not None:
                            self.current_target = self.rescue_center_pos
                    print(f"Return to {self.current_target}")
                elif self.current_target is not None and np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                    pass
                    # if self.nav.is_path_blocked(self.current_target):
                    #     print(f"Current target is blocked by wall!")
                    #     if self.pilot.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5): pass 
                    #     else:
                    #         bypass_node = self.nav.find_best_bypass(self.current_target)
                    #         if bypass_node is not None:
                    #             target_key = (int(self.current_target[0]), int(self.current_target[1]))
                    #             if target_key in self.nav.path_history:
                    #                 parent_of_blocked = self.nav.path_history[target_key]
                    #                 bypass_key = (int(bypass_node[0]), int(bypass_node[1]))
                    #                 self.nav.path_history[bypass_key] = parent_of_blocked
                    #                 self.current_target = bypass_node
                    #             else:
                    #                 self.current_target = bypass_node

        # --- DROPPING ---
        elif self.state == "DROPPING":
            self.drop_step += 1
            if self.drop_step > 100:
                self.state = "INITIAL"
                print(f"Switched to INITIAL")
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            
            if self.grasped_wounded_persons():
                return {"forward": -0.2, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            else:
                self.state = "INITIAL"
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- INITIAL ---
        elif self.state == "INITIAL":
            self.last_rescue_pos = None 
            self.initial_spot_pos = None 
            self.found_person_pos = None
            self.state = "COMMUTING"
            print(f"Switched to COMMUTING")
            self.current_target = None 
            return {"forward": -0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- COMMUTING ---
        elif self.state == "COMMUTING":
            print('Commuting to location before rescue')
            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                if len(self.nav.waypoint_stack) > 0:
                    next_waypoint = self.nav.waypoint_stack.pop()
                    self.current_target = next_waypoint
                else:
                    self.state = "EXPLORING"
                    print(f"Switched to EXPLORING")
                    self.current_target = self.position_before_rescue 
            
            elif np.linalg.norm(self.estimated_pos - self.current_target) > 30.0:
                pass
                # if self.nav.is_path_blocked(self.current_target):
                #     print(f"Current target is blocked by wall!")
                #     if self.pilot.is_blocked_by_drone(safety_dist=100.0, safety_angle=0.5): pass 
                #     else:
                #         bypass_node = self.nav.find_best_bypass(self.current_target)
                #         if bypass_node is not None: self.current_target = bypass_node
            
        # --- END_GAME ---
        elif self.state == "END_GAME":
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        return self.pilot.move_to_target_PID()

    def define_message_for_all(self):
        person_target = None
        if self.state in ["RESCUING", "RETURNING"]:
            if self.state == "RETURNING": person_target = self.last_rescue_pos
            else: person_target = self.current_target
        elif self.state == "EXPLORING":
            person_target = self.found_person_pos

        msg_data = {
            "id": self.identifier,
            "state": self.state,
            "person_pos": person_target, 
            "current_pos": self.estimated_pos,
            "priority": self.priority
        }
        return msg_data