import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math

class MyAgent:
    def __init__(self, num_agents: int) -> None:
        """
        Initialize the agent with the Q-learning algorithm.
        
        Args:
            num_agents: The number of agents in the environment.
        """
        self.num_agents = num_agents
        self.rng = np.random.default_rng()
        
        # Q-learning parameters - adjusted for better learning
        self.epsilon = 0.8  # Lower initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate as specified
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.alpha = 0.2  # Higher learning rate for faster updates
        self.gamma = 0.95  # Slightly lower discount factor to focus more on immediate rewards
        
        # Action space: 0=stay, 1=forward, 2=backward, 3=left, 4=right, 5=turn right, 6=turn left
        self.actions = list(range(7))
        
        # Centralized Q-table: Dictionary for sparse storage
        self.q_table = {}
        
        # Store last states, actions for updating
        self.last_states = [None] * num_agents
        self.last_actions = [None] * num_agents
        
        # Track statistics
        self.episode_count = 0
        self.total_updates = 0
        
    def discretize_state(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to discrete representation for Q-table lookup.
        Simplified discretization for better generalization.
        """
        # Extract info from state
        agent_x, agent_y = state[0], state[1]
        agent_o = state[2]
        goal_x, goal_y = state[4], state[5]
        
        # Calculate relative position to goal
        dx = goal_x - agent_x
        dy = goal_y - agent_y
        
        # Distance to goal (fewer bins)
        distance = math.sqrt(dx**2 + dy**2)
        distance_bin = min(int(distance / 5), 5)  # Coarser distance binning (6 bins)
        
        # Angle to goal relative to agent's orientation
        angle = math.atan2(dy, dx) - (agent_o * math.pi / 2)
        angle = (angle + 2*math.pi) % (2*math.pi)  # Normalize to [0, 2π)
        angle_bin = int(angle / (math.pi/2))  # 4 direction bins instead of 8
        
        # Process LIDAR data - more focused on obstacle presence than precise distance
        front_dist, front_type = state[6], state[7]
        right_dist, right_type = state[8], state[9] 
        left_dist, left_type = state[10], state[11]
        
        # Greatly simplify the state space for faster learning
        discrete_state = (
            distance_bin,
            angle_bin,
            1 if front_dist < 3 else 0,  # Binary: obstacle close in front?
            int(front_type > 0),         # Binary: any obstacle in front?
            int(right_dist < 2),         # Binary: obstacle close to right?
            int(left_dist < 2)           # Binary: obstacle close to left?
        )
        
        return discrete_state
    
    def get_action(self, states: np.ndarray, evaluation: bool = False) -> List[int]:
        """
        Select actions for all agents using epsilon-greedy policy.
        """
        actions = []
        
        for agent_id in range(self.num_agents):
            state = states[agent_id]
            
            # Skip if this agent is deactivated or evacuated
            if state[3] == 1 or state[3] == 2:  
                actions.append(0)  # Stay steady
                continue
            
            # Discretize state for Q-table lookup
            discrete_state = self.discretize_state(state)
            self.last_states[agent_id] = discrete_state
            
            # Epsilon-greedy action selection
            if not evaluation and self.rng.random() < self.epsilon:
                # Smart exploration: favor forward movement more often
                if self.rng.random() < 0.4:  # 40% chance of moving forward
                    action = 1  # forward
                else:
                    action = self.rng.choice(self.actions)
            else:
                # Exploitation: choose the best action from Q-table
                if discrete_state not in self.q_table:
                    self.q_table[discrete_state] = np.zeros(len(self.actions))
                
                # Get the best action (or random among ties)
                q_values = self.q_table[discrete_state]
                best_actions = np.where(q_values == np.max(q_values))[0]
                action = self.rng.choice(best_actions)
            
            # Store action for later update
            self.last_actions[agent_id] = action
            actions.append(action)
        
        return actions
    
    def update_policy(self, actions: List[int], next_states: np.ndarray, rewards: np.ndarray) -> None:
        """
        Update the Q-table based on the rewards received.
        """
        self.total_updates += 1
        
        for agent_id in range(self.num_agents):
            # Skip if no previous state (first step)
            if self.last_states[agent_id] is None:
                continue
            
            last_state = self.last_states[agent_id]
            action = self.last_actions[agent_id]
            reward = rewards[agent_id]
            
            # Initialize Q-value for last state if not present
            if last_state not in self.q_table:
                self.q_table[last_state] = np.zeros(len(self.actions))
            
            # Special case for evacuated/deactivated agents
            next_state = next_states[agent_id]
            if next_state[3] == 1 or next_state[3] == 2:
                current_q = self.q_table[last_state][action]
                self.q_table[last_state][action] = current_q + self.alpha * (reward - current_q)
                continue
            
            # Get the next state's discretized representation
            next_discrete_state = self.discretize_state(next_state)
            
            # Initialize next state if not in Q-table
            if next_discrete_state not in self.q_table:
                self.q_table[next_discrete_state] = np.zeros(len(self.actions))
            
            # Calculate target Q-value using the Bellman equation
            max_next_q = np.max(self.q_table[next_discrete_state])
            target = reward + self.gamma * max_next_q
            
            # Update Q-value for the last state-action pair
            current_q = self.q_table[last_state][action]
            self.q_table[last_state][action] = current_q + self.alpha * (target - current_q)
    
    def decay_epsilon(self) -> None:
        """Reduce exploration rate after each episode and print statistics."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Print debug info every few episodes
        if self.episode_count % 10 == 0:
            print(f"Episode {self.episode_count}: epsilon={self.epsilon:.3f}, Q-table size={len(self.q_table)}")
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.last_states = [None] * self.num_agents
        self.last_actions = [None] * self.num_agents