import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


class CommunicatorHandler:
    def __init__(self, drone):
        self.drone = drone

    # The function should_wait_in_queue() was removed
    # to completely eliminate the old queue-based waiting mechanism

    def is_target_taken_or_better_candidate(self, target_person_pos):
        # If there is no target, nothing to coordinate
        if target_person_pos is None:
            return False

        # If communication is disabled, we cannot coordinate with teammates
        if self.drone.communicator_is_disabled():
            return False
        
        # Distance from this drone to the target person
        my_dist = np.linalg.norm(self.drone.estimated_pos - target_person_pos)

        # Threshold to consider two target coordinates as the same person
        COORDINATE_MATCH_THRESHOLD = 50.0 

        # Iterate through all received messages from other drones
        for msg_package in self.drone.communicator.received_messages:

            # Extract message content (handle both tuple and raw dict formats)
            content = msg_package[1] if isinstance(msg_package, tuple) else msg_package
            if not isinstance(content, dict):
                continue
            
            # Information shared by the other drone
            other_id = content.get("id")
            other_state = content.get("state")
            other_person_pos = content.get("person_pos")
            other_current_pos = content.get("current_pos")
            
            # Skip invalid or incomplete messages
            if other_person_pos is None or other_current_pos is None:
                continue
            
            # Distance between both drones' targeted persons
            dist_between_targets = np.linalg.norm(target_person_pos - other_person_pos)
            
            # --------------------------------------------------
            # If both drones are aiming at the SAME victim
            # --------------------------------------------------
            if dist_between_targets < COORDINATE_MATCH_THRESHOLD:
                
                # Distance from the other drone to the victim
                other_dist_to_person = np.linalg.norm(other_current_pos - other_person_pos)

                # --------------------------------------------------
                # CASE 1: The other drone has already secured the victim
                # (RETURNING or DROPPING)
                # -> Give up immediately
                # --------------------------------------------------
                if other_state in ["RETURNING", "DROPPING"]:
                    return True 

                # --------------------------------------------------
                # CASE 2: Both drones are competing for the same victim
                # (RESCUING vs RESCUING, EXPLORING vs RESCUING, etc.)
                #
                # Strategy:
                #   1. Compare distances
                #   2. Use ID as tie-breaker
                # --------------------------------------------------
                
                # If the opponent is significantly closer (> 20 px) -> yield
                if other_dist_to_person < my_dist - 20.0:
                    return True
                
                # If distances are similar (within 20 px) -> compare IDs
                # Smaller ID gets priority (simple deterministic rule
                # to break deadlocks and avoid oscillation)
                if abs(other_dist_to_person - my_dist) <= 20.0:
                    if other_id < self.drone.identifier:
                        return True

        # No better candidate found -> we can continue targeting
        return False
