import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)
    
    # Convert goal_area to a list of tuples for easier checking
    goal_area_tuples = [tuple(g) for g in goal_area]
    goal_center = np.mean(goal_area, axis=0)

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            # Already evacuated agents get no additional reward
            continue
        elif i in deactivated_agents:
            # Significant penalty for deactivation (collision)
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area_tuples:
            # Substantial reward for reaching goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Calculate the distances to goal before and after the move
            old_distance = min(np.linalg.norm(np.array(old_pos) - np.array(goal)) for goal in goal_area)
            new_distance = min(np.linalg.norm(np.array(new_pos) - np.array(goal)) for goal in goal_area)
            
            # Base reward with small penalty to encourage efficiency
            rewards[i] = -0.1
            
            # Significant reward for getting closer to goal
            distance_delta = old_distance - new_distance
            if distance_delta > 0:
                # Reward proportional to progress
                rewards[i] += 5.0 * distance_delta
            else:
                # Small penalty for moving away
                rewards[i] += 1.0 * distance_delta  # This will subtract since delta is negative
            
            # Penalty for staying in place
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 1.0
                
            # Add a small directional reward (closer to goal center is better)
            direction_to_goal = np.array(goal_center) - np.array(new_pos)
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal) if np.linalg.norm(direction_to_goal) > 0 else np.zeros(2)
            movement_direction = np.array(new_pos) - np.array(old_pos)
            movement_direction = movement_direction / np.linalg.norm(movement_direction) if np.linalg.norm(movement_direction) > 0 else np.zeros(2)
            
            # Dot product to measure alignment of movement with direction to goal
            if np.linalg.norm(movement_direction) > 0:
                alignment = np.dot(direction_to_goal, movement_direction)
                rewards[i] += 2.0 * alignment
                
    return rewards, evacuated_agents