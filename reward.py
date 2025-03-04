import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)
    
    # Convert goal_area to a list of tuples for easier checking
    goal_area_tuples = [tuple(g) for g in goal_area]

    # Compute reward for each agent
    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            # Already evacuated agents get no additional reward
            continue
        elif i in deactivated_agents:
            # Penalty for deactivation
            rewards[i] = -100.0
        elif tuple(new_pos) in goal_area_tuples:
            # Substantial reward for reaching goal
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            # Small step penalty to encourage efficiency
            rewards[i] = -0.1
            
            # Calculate the distances to goal before and after the move
            old_distance = min(np.linalg.norm(np.array(old_pos) - np.array(goal)) for goal in goal_area)
            new_distance = min(np.linalg.norm(np.array(new_pos) - np.array(goal)) for goal in goal_area)
            
            # Reward for moving closer to the goal, penalty for moving away
            distance_reward = 2.0 * (old_distance - new_distance)
            rewards[i] += distance_reward
            
            # Check if agent stayed in place (penalize slightly)
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 0.5
                
            # Limit the minimum reward
            rewards[i] = max(rewards[i], -5.0)

    return rewards, evacuated_agents