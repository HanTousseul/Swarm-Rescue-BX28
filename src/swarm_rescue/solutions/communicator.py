import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from .mapping import THRESHOLD_MIN, THRESHOLD_MAX

MAPPING_REFRESH_RATE = 100 # in timesteps, time between updates to the map by the same drone

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.comm_counter = 0 
        self.map_date_update = [0 for i in range (10)] #timestep of last map update given by a certain drone
        self.list_wounded = []
        self.everyone_home = False

    def process_incoming_messages(self) -> None:
        '''
        Takes care of receiving messages, and assigning all interesting data to corresponding variables
        
        :param self: self
        '''

        self.drone.priority = self.get_priority()
        self.drone.has_priority = False
        drone_with_bigger_priority:bool = False

        nb_drones_returned = 0

        self.list_nearby_drones = []
        self.list_vip_drones = [] # [NEW] Track drones carrying victims (Priority yielding)
        self.list_victims_taken_care_of = []
        self.list_received_maps = [None for i in range(10)] #list_received_map[n] = map given by drone whose identifier is n
        if self.drone.communicator_is_disabled(): return

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            sender_id = content.get("id")
            if sender_id == self.drone.identifier: continue 

            drone_id = content['id']

            # [UPDATED] Separate drones based on their 'grasping' state
            is_grasping = content.get('grasping', False)
            if is_grasping:
                self.list_vip_drones.append(content['position'])
            else:
                self.list_nearby_drones.append(content['position'])

            if content['victim_chosen'] is not None:
                dist = content.get('victim_chosen_dist', float('inf'))
                self.list_victims_taken_care_of.append((content['victim_chosen'], dist, content['id']))

            if self.drone.cnt_timestep - self.map_date_update[content['id']] > MAPPING_REFRESH_RATE:
                self.list_received_maps[drone_id] = content['obstacle_map']

            if content['priority'] > self.drone.priority: 
                drone_with_bigger_priority = True
                self.drone.has_priority = False

            if self.drone.state == 'END_GAME' and self.drone.is_inside_return_area and content['returned']:

                nb_drones_returned += 1

            if content['everyone_home']: self.everyone_home = True

        # Syncronize rescued person list
        global_rescued = []
        # global_victims = []
        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if isinstance(content, dict):
                if 'rescued_victims' in content:
                    global_rescued.extend(content['rescued_victims'])
                # if 'victim_list' in content:
                #     global_victims.extend(content['victim_list'])
                
        self.drone.victim_manager.sync_rescued_victims(global_rescued)
        # self.drone.victim_manager.sync_registry(global_victims)

        # Clear rescued victim fron list
        valid_taken_care_of = []
        for v_pos, v_dist, v_id in self.list_victims_taken_care_of:
            if not any(np.linalg.norm(v_pos - r_pos) < 30.0 for r_pos in self.drone.victim_manager.rescued_victims):
                valid_taken_care_of.append((v_pos, v_dist, v_id))
                
        self.list_victims_taken_care_of = valid_taken_care_of
        
        # print('nb_drones_returned',self.drone.identifier, nb_drones_returned, self.everyone_home)
        if nb_drones_returned == len(self.drone.communicator.received_messages) and not self.everyone_home: 
            print(f'{self.drone.identifier} everyone home')
            self.everyone_home = True
        else: self.everyone_home = False
        if not drone_with_bigger_priority: self.has_priority = True

        self.consolidate_maps()
        return          

    def consolidate_maps(self) -> None:

        '''
        This function will handle consolidating maps from other drones. Concretely, it will receive maps from nearby drones, make sure that we haven't received an update in a while and applies the update if necessary to the map.
        
        :param self: self
        :return: None
        :rtype: None
        '''
        OBSTACLE_THRESHOLD = 5.0
        FREE_SPACE_THRESHOLD = -1.0
        for drone_id in range(10):

            obs_map = self.list_received_maps[drone_id]
            if obs_map is None:
                continue
            # Obstacle updates (received > current)
            mask_obstacle = (obs_map > OBSTACLE_THRESHOLD) & (obs_map > self.drone.nav.obstacle_map.grid)
            self.drone.nav.obstacle_map.grid[mask_obstacle] = obs_map[mask_obstacle]

            # Free space updates (received < current, both negative)
            mask_free = (obs_map < FREE_SPACE_THRESHOLD) & (obs_map < self.drone.nav.obstacle_map.grid)
            self.drone.nav.obstacle_map.grid[mask_free] = obs_map[mask_free]

            self.map_date_update[drone_id] = self.drone.cnt_timestep
        self.drone.nav.obstacle_map.update_cost_map()
    

    def get_priority(self) -> int: 
        '''
        Returns an integer giving the "priority" of a drone, the higher the integer, the more priority the drone gets. this helps resolve traffic jams. if a drone is carrying a wounded, it has the priority over non-carrying drones. in case of a tie, the drone with the highest identifier wins.
        
        :param self: self
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
        if self.drone.state == 'RESCUING':
            victim = self.drone.current_target
        elif self.drone.state == 'EXPLORING':
            victim = self.drone.current_target_best_victim_pos
        else: 
            victim = None

        dist_to_victim = float('inf')
        if victim is not None:
            dist_to_victim = float(np.linalg.norm(self.drone.estimated_pos - victim))

        if self.drone.cnt_timestep % 51 == self.drone.identifier * 5:
            # print('Sent map!')
            obstacle_map = self.drone.nav.obstacle_map.grid  
        else: obstacle_map = None

        if self.drone.state == 'END_GAME' and self.drone.is_inside_return_area:

            returned = True

        else: returned = False

        return_dict =   {'id' : self.drone.identifier,
                        'position' : self.drone.estimated_pos,
                        'state' : self.drone.state,
                        'grasping' : True if self.drone.grasped_wounded_persons() else False,
                        'obstacle_map' : obstacle_map,
                        'priority': self.get_priority(),
                        # 'victim_list': self.drone.victim_manager.registry if self.drone.cnt_timestep % 10 == self.drone.identifier else [],
                        'rescued_victims': self.drone.victim_manager.rescued_victims if self.drone.cnt_timestep % 10 == self.drone.identifier else [],
                        'victim_chosen': victim if self.drone.cnt_timestep % 10 == self.drone.identifier else None,
                        'victim_chosen_dist': dist_to_victim if self.drone.cnt_timestep % 10 == self.drone.identifier else None,
                        'returned': returned,
                        'everyone_home': self.everyone_home
        }

        return return_dict