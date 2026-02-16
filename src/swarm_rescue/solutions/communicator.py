import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from .mapping import THRESHOLD_MIN, THRESHOLD_MAX

class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone
        self.comm_counter = 0 
        self.map_date_update = [0 for i in range (10)] #timestep of last map update given by a certain drone
        
    def process_incoming_messages(self):
        if self.drone.communicator_is_disabled(): return

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue
            
            sender_id = content.get("id")
            if sender_id == self.drone.identifier: continue 

            msg_type = content.get("type")
            
            # 1. Nhận tin XÍ CHỖ
            if msg_type == "CLAIM_TARGET":
                pos = content.get("pos")
                if pos is not None:
                    # Lưu vào danh sách bận của Driver (Hiệu lực 6-7 giây)
                    self.drone.busy_targets.append({
                        "pos": np.array(pos),
                        "timer": 400 
                    })

            # 2. Nhận tin OBSTACLE (Vẽ lên bản đồ mình)
            elif msg_type == "MAP_OBSTACLE":
                points = content.get("points")
                if points:
                    self.drone.nav.obstacle_map.update_from_remote_points(points)

            # 3. Nhận tin CLEAR ZONE
            elif msg_type == "CLEAR_ZONE":
                clear_pos = content.get("pos")
                if clear_pos is not None:
                    # Gọi hàm clear_area với bán kính lớn (5)
                    self.drone.nav.victim_map.clear_area(np.array(clear_pos), radius_grid=5)
            
            # 4. Nhận tin VICTIM
            elif msg_type == "MAP_UPDATE" and content.get("victim_found"):
                victim_pos = content.get("victim_pos")
                if victim_pos is not None:
                    gx, gy = self.drone.nav.victim_map.world_to_grid(victim_pos[0], victim_pos[1])
                    self.drone.nav.victim_map.grid[gy, gx] += 5.0

    def incoming_messages_maps(self) -> None:

        '''
        This function will handle receiving and consolidating maps from other drones. Concretely, it will receive maps from nearby drones, make sure that we haven't received an update in a while and applies the update if necessary to both of the maps.
        
        :param self: self
        '''
        MAP_REFRESH_RATE = 200 # in timesteps, time between updates to the map by the same drone

        list_received_maps = [None for i in range(10)] #list_received_map[n] = map given by drone whose identifier is n
        if self.drone.communicator_is_disabled(): return

        for msg_package in self.drone.communicator.received_messages:
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict): continue

            drone_id = content['id']

            if self.drone.cnt_timestep - self.map_date_update[content['id']] > MAP_REFRESH_RATE:

                list_received_maps[drone_id] = content['obstacle_map']

        for obs_map in list_received_maps:

            if obs_map is None: continue
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
    
    def create_new_message(self) -> dict:
        '''
        This function handles the creation of the message to be communicated by the drone
        we only send the map every 50 timestep to optimise the code further
        
        :param self: self
        :return: All information necessary to be communicated
        :rtype: dict
        '''
        
        return_dict =   {'id' : self.drone.identifier,
                        'position' : self.drone.estimated_pos,
                        'state' : self.drone.state,
                        'grasping' : True if self.drone.grasped_wounded_persons() else False,
                        'obstacle_map' : self.drone.nav.obstacle_map.grid if self.drone.cnt_timestep % 50 == 0 else None
        }

        return return_dict