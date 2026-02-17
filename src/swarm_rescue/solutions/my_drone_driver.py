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
        
        self.last_rescue_pos = None
        self.initial_spot_pos = None
        self.found_person_pos = None
        self.patience = None
        self.not_grasped = False
        self.drop_step = 0

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
        
        # [INTEGRATION] Update Dead Drone Memory at start of frame
        self.comms.other_pos()

        check_center = False
        
        # 1. Update Navigator & Sensors
        self.nav.update_navigator()
        REACH_THRESHOLD_LOCAL = 24.0 if self.grasped_wounded_persons() else 35.0
        
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
                
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    check_center = True
                    if self.rescue_center_pos is None:
                       self.rescue_center_pos = np.array([obj_x, obj_y])
            
            if closest_person_pos is not None:
                # [INTEGRATION] Check if target is taken or if it is in a "Dead Zone"
                if not self.comms.is_target_taken_or_better_candidate(closest_person_pos) and \
                   not self.comms.is_forbidden(closest_person_pos):
                    ALPHA = 0.7
                    if self.found_person_pos is None:
                        self.found_person_pos = closest_person_pos
                    else:
                        self.found_person_pos = ALPHA * closest_person_pos + (1 - ALPHA) * self.found_person_pos

        # 3. Force return when remaining timestep is low
        steps_remaining = self.max_timesteps - self.cnt_timestep
        RETURN_TRIGGER_STEPS = int(self.max_timesteps * 0.2)
        
        if steps_remaining <= RETURN_TRIGGER_STEPS:
            if self.is_inside_return_area: self.state = "END_GAME"
            else:
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
                            self.current_target = self.rescue_center_pos if self.rescue_center_pos is not None else self.initial_position
                    
                    self.state = "RETURNING"
                    self.not_grasped = True

        # 4. HARD STUCK DETECTION & UNSTICK
        self.pos_history_long.append(self.estimated_pos.copy())
        if len(self.pos_history_long) > 60:
            self.pos_history_long.pop(0) 
        
        if len(self.pos_history_long) == 60 and (steps_remaining > RETURN_TRIGGER_STEPS or not self.is_inside_return_area):
            start_pos = self.pos_history_long[0]
            dist_moved = np.linalg.norm(self.estimated_pos - start_pos)
            
            if dist_moved < 17.0:
                lat_force = 1.0 if random.random() > 0.5 else -1.0
                if self.current_target is not None:
                    d_x = self.current_target[0] - self.estimated_pos[0]
                    d_y = self.current_target[1] - self.estimated_pos[1]
                    target_angle = math.atan2(d_y, d_x)
                    angle_diff = (target_angle - self.estimated_angle + math.pi) % (2 * math.pi) - math.pi
                    lat_force = 1.0 if angle_diff > 0 else -1.0

                return {
                    "forward": -0.7, 
                    "lateral": lat_force,
                    "rotation": 0.0,
                    "grasper": 1 if self.grasped_wounded_persons() else 0
                }

        # 5. STATE MACHINE
        # --- EXPLORING ---
        if self.state == "EXPLORING":
            if self.found_person_pos is not None:
                if self.comms.is_target_taken_or_better_candidate(self.found_person_pos):
                    self.nav.visit(self.found_person_pos) 
                    self.found_person_pos = None 
                else:
                    self.state = "RESCUING"
                    self.position_before_rescue = self.current_target if self.current_target is not None else self.estimated_pos
                    self.current_target = self.found_person_pos
                    self.initial_spot_pos = self.found_person_pos.copy()
            
            elif self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                if self.current_target is not None: self.nav.visit(self.current_target)
                else: self.nav.visit(self.estimated_pos)
                self.nav.update_mapper()
                
                pos_key = (int(self.estimated_pos[0]), int(self.estimated_pos[1]))
                next_target = None
                
                if pos_key in self.nav.edge and len(self.nav.edge[pos_key]) > 0:
                    while len(self.nav.edge[pos_key]) > 0:
                        candidate = np.array(self.nav.edge[pos_key].pop())
                        
                        # [INTEGRATION] Offset candidate if it is inside a dead zone
                        candidate = self.comms.avoid_forbidden_target(candidate)
                        
                        is_safe = True
                        if self.rescue_center_pos is not None:
                            dist_to_rc = np.linalg.norm(candidate - self.rescue_center_pos)
                            if dist_to_rc < 150.0: is_safe = False
                        if self.nav.is_path_blocked(candidate, 5): is_safe = False
                        
                        if is_safe:
                            next_target = candidate
                            break
                
                if next_target is not None:
                    target_int_key = (int(next_target[0]), int(next_target[1]))
                    self.nav.path_history[target_int_key] = self.current_target.copy() if self.current_target is not None else self.estimated_pos.copy()
                    self.current_target = next_target
                else:
                    if self.current_target is not None:
                        current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                        self.current_target = self.nav.path_history.get(current_int_key, self.estimated_pos.copy())

        # --- RESCUING ---
        elif self.state == "RESCUING":
            if self.current_target is not None:
                if self.comms.is_target_taken_or_better_candidate(self.current_target):
                    self.state = "EXPLORING"
                    self.found_person_pos = None
                    self.current_target = self.position_before_rescue
                    return self.pilot.move_to_target_PID()

            if not self.grasped_wounded_persons():
                self.patience = (self.patience or 0) + 1
                if self.patience > 80:
                    self.state = "EXPLORING"
                    self.current_target = self.position_before_rescue
                    self.found_person_pos = None
                    self.patience = None
                    return self.pilot.move_to_target_PID()

            if self.grasped_wounded_persons():
                self.last_rescue_pos = self.current_target.copy()
                self.state = "RETURNING"
                self.current_target = self.position_before_rescue if self.position_before_rescue is not None else self.rescue_center_pos

        # --- RETURNING ---
        elif self.state == "RETURNING":
            if check_center and self.rescue_center_pos is not None:
                if not self.grasped_wounded_persons(): self.state = "DROPPING"
                self.current_target = self.rescue_center_pos
                if np.linalg.norm(self.estimated_pos - self.current_target) < 10.0:
                    self.state = "DROPPING"
                    self.drop_step = 0
            else:
                if self.current_target is not None and np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                    if np.linalg.norm(self.current_target - self.initial_position) < 10:
                        self.state = 'END_GAME'
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
                    
                    current_int_key = (int(self.current_target[0]), int(self.current_target[1]))
                    if current_int_key in self.nav.path_history: 
                        parent_node = self.nav.path_history[current_int_key]
                        self.nav.waypoint_stack.append(self.current_target.copy())
                        self.current_target = parent_node 
                    else:
                        self.current_target = self.rescue_center_pos if self.rescue_center_pos is not None else self.initial_position

        # --- DROPPING ---
        elif self.state == "DROPPING":
            self.drop_step += 1
            if self.drop_step > 100 or not self.grasped_wounded_persons():
                self.state = "INITIAL"
                return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            return {"forward": -0.2, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- INITIAL ---
        elif self.state == "INITIAL":
            self.last_rescue_pos = None 
            self.initial_spot_pos = None 
            self.found_person_pos = None
            self.state = "COMMUTING"
            self.current_target = None 
            return {"forward": -0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # --- COMMUTING ---
        elif self.state == "COMMUTING":
            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < REACH_THRESHOLD_LOCAL:
                if len(self.nav.waypoint_stack) > 0:
                    self.current_target = self.nav.waypoint_stack.pop()
                    # [INTEGRATION] Check if commuting waypoint is now forbidden
                    self.current_target = self.comms.avoid_forbidden_target(self.current_target)
                else:
                    self.state = "EXPLORING"
                    self.current_target = self.position_before_rescue 

        # --- END_GAME ---
        elif self.state == "END_GAME":
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        return self.pilot.move_to_target_PID()

    def define_message_for_all(self):
        person_target = None
        if self.state in ["RESCUING", "RETURNING"]:
            person_target = self.last_rescue_pos if self.state == "RETURNING" else self.current_target
        elif self.state == "EXPLORING":
            person_target = self.found_person_pos
        
        # Collect all forbidden zones (lost drones + detected stuck drones)
        forbidden_list = [pos.tolist() for pos in self.comms.forbidden.values()]
        return {
            "id": self.identifier,
            "state": self.state,
            "person_pos": person_target, 
            "current_pos": self.estimated_pos,
            "forbidden_zones": forbidden_list
        }