import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
try:
    from .navigator import Navigator
except ImportError:
    from navigator import Navigator
from .mapping import THRESHOLD_MIN, THRESHOLD_MAX

MAPPING_REFRESH_RATE = 200 # in timesteps, time between updates to the map by the same drone

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.comm_counter = 0 
        self.map_date_update = [0 for i in range (10)] # timestep of last map update given by a certain drone
        self.list_wounded = []
        
        # --- ZONES LOGIC STORAGE ---
        self.nav = Navigator(self)
        self.forbidden = dict()          # Unified storage (IDs and 'stuck_' keys)
        self.other_drones_pos = dict()   # Dictionary of {id: last_known_pos}
        self.FORBIDDEN_RADIUS = 100
        self.INERTIA = 50

    def process_incoming_messages(self) -> None:
        self.list_nearby_drones = []
        self.list_received_maps = [None for i in range(10)]
        if self.drone.communicator_is_disabled(): return

        # Update tracking of silent and stationary drones
        self.update_forbidden_zones()

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            sender_id = content.get("id")
            if sender_id is None or sender_id == self.drone.identifier: continue 

            drone_id = content['id']
            self.list_nearby_drones.append(content['position']) 

            # Sync stationary drone coordinates shared by other drones
            remote_zones = content.get("forbidden_zones", [])
            for zone_pos in remote_zones:
                rk = f"stuck_{int(zone_pos[0]/5)}_{int(zone_pos[1]/5)}"
                if rk not in self.forbidden:
                    self.forbidden[rk] = np.array(zone_pos)

            # --- MAPPING LOGIC ---
            if self.drone.cnt_timestep - self.map_date_update[drone_id] > MAPPING_REFRESH_RATE:
                self.list_received_maps[drone_id] = content.get('obstacle_map')

            # --- VICTIM LOGIC ---
            for elt in content.get('victim_list', []):
                if elt not in self.drone.victim_manager.registry:
                   self.drone.victim_manager.registry.append(elt)
        
        self.consolidate_maps()
        return         
    
    def other_pos(self):
        """Save other drones' position and detect lost drones safely. In addition,
        modify self.forbidden if a drone marked as dead has comms again.
        """
        if not hasattr(self, "other_drones_pos") or self.other_drones_pos is None:
            self.other_drones_pos = {}
    
        # Initialize forbidden dictionary if not exists
        if not hasattr(self, "forbidden") or self.forbidden is None:
            self.forbidden = {}  # key: drone_id, value: last-known position
    
        # Create a safe list of IDs we currently know
        id_list = list(self.other_drones_pos.keys())
    
        # Check if communicator is available and has messages
        if hasattr(self.drone, "communicator") and hasattr(self.drone.communicator, "received_messages"):
            msgs = self.drone.communicator.received_messages
            if msgs:
                for msg in msgs:
                    if msg and isinstance(msg, dict) and "id" in msg and "current_pos" in msg:
                        drone_id = msg["id"]
                        if drone_id == self.drone.identifier:
                            continue
                        pos = np.array(msg["current_pos"])
                        self.other_drones_pos[drone_id] = pos
                        if drone_id in id_list:
                            id_list.remove(drone_id)
                        # Remove from forbidden if drone is back online
                        if drone_id in self.forbidden:
                            del self.forbidden[drone_id]
    
        # Any drone ID left in id_list has lost comms → mark as forbidden
        for drone_id in id_list:
            pos = self.other_drones_pos.get(drone_id)
            if pos is not None:
                self.forbidden[drone_id] = [pos]  # store last-known position

    def mark_as_stuck(self):
        """
        Handles current stationary positions for drones physically detected as 
        stuck. 
        """
        if not hasattr(self.drone, 'nav'): return
        
        coords_list = self.drone.nav.get_stationary_drone_coordinates()
        if not coords_list: return
        
        for pos in coords_list:
            for (drone_id, coord) in self.forbidden:
                if (abs(pos[0] - coord[0]) ** 2 + abs(pos[1] - coord[1]) ** 2 >= self.INERTIA ** 2):
                    self.forbidden[drone_id].append(np.array(pos)) # Logic: lost comms + stuck = dead
            
    
    def update_forbidden_zones(self):
        """
        Combined logic to return:
        1. Last-known position of drones that lost communication.
        2. Stationary drone positions from the navigator.
        3. Recovery: Removes drones from forbidden if they start talking again.
        """
        # Update from Comms
        self.other_pos()
        
        # Update from Sensors
        self.mark_as_stuck()
        
        return self.forbidden

    def consolidate_maps(self) -> None:

        '''
        This function will handle consolidating maps from other drones. Concretely, it will receive maps from nearby drones, make sure that we haven't received an update in a while and applies the update if necessary to the map.
        
        :param self: self
        :return: None
        :rtype: None
        '''

        for drone_id in range(10):

            obs_map = self.list_received_maps[drone_id]
            if obs_map is None:
                continue
            for y in range(len(obs_map)):
                for x in range(len(obs_map[y])):

                    diff = obs_map[y][x] - self.drone.nav.obstacle_map.grid[y][x]

                    if obs_map[y][x] > 0:

                        if diff <= 0: continue
                        self.drone.nav.obstacle_map.grid[y][x] = obs_map[y][x]

                    else:
                        if diff >= 0: continue
                        self.drone.nav.obstacle_map.grid[y][x] = obs_map[y][x]

                    self.drone.nav.obstacle_map.grid[y][x] = obs_map[y][x]

            self.map_date_update[drone_id] = self.drone.cnt_timestep
    

    def priority(self) -> int: 
        '''
        Returns an integer giving the "priority" of a drone, 
        the higher the integer, the more priority the drone gets. this helps resolve traffic jams. 
        If a drone is carrying a wounded, it has the priority over non-carrying drones. 
        In case of a tie, the drone with the highest identifier wins.
        :return: Priority value
        :rtype: int
        '''
        
        return self.drone.identifier + 100 if self.drone.grasped_wounded_persons() else self.drone.identifier        


    def create_new_message(self) -> dict:
        '''
        This function handles the creation of the message to be communicated by the drone
        we only send the map every 50 timestep to optimise the code further:
        'id',
        'position',
        'state',
        'obstacle_map' ,
        'priority',
        'victim_list',
        'victim_chosen',
        
        :param self: self
        :return: All information necessary to be communicated
        :rtype: dict
        '''
        if self.drone.state == 'EXPLORING':

            victim = self.drone.best_victim_pos
        
        else: victim = None
        if self.drone.cnt_timestep % 51 == self.drone.identifier * 5:
            obstacle_map = self.drone.nav.obstacle_map.grid  
        else: obstacle_map = None
        
        # List of all forbidden coordinates (stuck + lost comms)
        forbidden_list = [pos.tolist() for pos in self.forbidden.values()]
        
        return_dict =   {'id' : self.drone.identifier,
                        'position' : self.drone.estimated_pos,
                        'state' : self.drone.state,
                        'grasping' : True if self.drone.grasped_wounded_persons() else False,
                        'obstacle_map' : obstacle_map,
                        'priority': self.priority(),
                        'victim_list': self.drone.victim_manager.registry,
                        'victim_chosen': victim,
                        'forbidden_zones': forbidden_list
        }

        return return_dict