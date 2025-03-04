import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MyAgent():
    def __init__(self, num_agents: int):
        # Parameters
        self.num_agents = num_agents
        self.n_actions = 7  # 0:stay, 1-4:movement, 5-6:rotation
        
        # Important: Define exact state dimension
        self.state_dim = 22  # Fixed dimension after feature extraction
        
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory_size = 50000
        self.update_target_freq = 1000
        self.training_steps = 0
        
        # Create networks for each agent
        self.policy_nets = []
        self.target_nets = []
        self.optimizers = []
        self.replay_buffers = []
        
        for i in range(num_agents):
            # Policy network (for action selection)
            policy_net = DQN(self.state_dim, self.n_actions).to(self.device)
            
            # Target network (for stable Q-value targets)
            target_net = DQN(self.state_dim, self.n_actions).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            
            # Optimizer
            optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
            
            # Experience replay buffer
            replay_buffer = deque(maxlen=self.memory_size)
            
            self.policy_nets.append(policy_net)
            self.target_nets.append(target_net)
            self.optimizers.append(optimizer)
            self.replay_buffers.append(replay_buffer)
        
        # Store previous states and actions for learning
        self.previous_states = [None] * num_agents
        self.previous_actions = [None] * num_agents
        self.loss_criterion = nn.MSELoss()
        
    def _state_to_tensor(self, state):
        """Convert state array to tensor with consistent feature engineering"""
        # Handle terminal states or invalid states
        if state is None or (isinstance(state, np.ndarray) and state[0] == -1):
            # Return zeros for terminal states
            return torch.zeros(self.state_dim, device=self.device)
        
        # Basic features
        pos_x, pos_y = float(state[0]), float(state[1])
        orientation = int(state[2])
        goal_x, goal_y = float(state[4]), float(state[5])
        
        # Calculate relative goal position
        rel_x = goal_x - pos_x
        rel_y = goal_y - pos_y
        
        # Distance to goal
        goal_dist = np.sqrt(rel_x**2 + rel_y**2)
        
        # Orientation features (one-hot)
        orientation_one_hot = [0.0, 0.0, 0.0, 0.0]
        if 0 <= orientation < 4:  # Safety check
            orientation_one_hot[orientation] = 1.0
        
        # Process LIDAR data
        lidar_features = []
        for i in range(6, 24, 2):  # Process all 9 LIDAR directions
            if i < len(state):
                dist = float(state[i])
                obj_type = float(state[i+1])
                # Normalize distance (1.0 for max range, less for closer objects)
                norm_dist = min(dist / 10.0, 1.0)
                lidar_features.extend([norm_dist, obj_type])
        
        # Ensure we have the right number of LIDAR features
        while len(lidar_features) < 18:
            lidar_features.extend([0.0, 0.0])  # Pad with zeros if needed
        
        # Combine all features
        features = [
            pos_x / 50.0,  # Normalize positions
            pos_y / 50.0,
            rel_x / 50.0,
            rel_y / 50.0,
            goal_dist / 70.0,  # Normalize distance
            *orientation_one_hot,
            *lidar_features[:18]  # Take exactly 18 LIDAR features
        ]
        
        # Ensure we have exactly state_dim features
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
        
    def get_action(self, states, evaluation=False):
        """Choose actions using epsilon-greedy policy"""
        actions = []
        
        for i in range(self.num_agents):
            state = states[i]
            
            # Skip deactivated or evacuated agents (-1 values)
            if state[0] == -1:
                actions.append(0)  # Stay in place
                self.previous_states[i] = None  # Mark as no state
                self.previous_actions[i] = 0
                continue
            
            try:
                # Convert state to tensor
                state_tensor = self._state_to_tensor(state)
                
                # Epsilon-greedy action selection
                if evaluation or random.random() > self.epsilon:
                    # Exploitation: choose best action
                    with torch.no_grad():
                        q_values = self.policy_nets[i](state_tensor.unsqueeze(0))
                        action = q_values.argmax().item()
                else:
                    # Exploration: choose random action
                    action = random.randint(0, self.n_actions - 1)
                
                # Store current state and action for learning
                self.previous_states[i] = state
                self.previous_actions[i] = action
                
                actions.append(action)
            except Exception as e:
                print(f"Error in get_action for agent {i}: {e}")
                actions.append(0)  # Default action in case of error
                self.previous_states[i] = None
        
        return actions

    def update_policy(self, actions, new_states, rewards):
        """Update policy using experience replay and DQN"""
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store transitions
        for i in range(self.num_agents):
            if self.previous_states[i] is None:
                continue
                
            # Check if the agent is deactivated or reached goal
            is_terminal = new_states[i][0] == -1
            
            # Store transition
            self.replay_buffers[i].append((
                self.previous_states[i],
                self.previous_actions[i],
                rewards[i],
                None if is_terminal else new_states[i],
                is_terminal
            ))
        
        # Train on mini-batches
        self.training_steps += 1
        
        for i in range(self.num_agents):
            # Skip training if we don't have enough samples
            if len(self.replay_buffers[i]) < self.batch_size:
                continue
            
            try:
                # Sample a mini-batch
                mini_batch = random.sample(self.replay_buffers[i], self.batch_size)
                
                # Process batch data
                batch_states = []
                batch_actions = []
                batch_rewards = []
                batch_next_states = []
                batch_terminals = []
                
                # Process each experience in the batch
                for s, a, r, ns, t in mini_batch:
                    batch_states.append(self._state_to_tensor(s))
                    batch_actions.append(a)
                    batch_rewards.append(r)
                    batch_next_states.append(self._state_to_tensor(ns))
                    batch_terminals.append(t)
                
                # Convert to tensors
                state_tensor = torch.stack(batch_states)
                action_tensor = torch.tensor(batch_actions, dtype=torch.int64, device=self.device)
                reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
                next_state_tensor = torch.stack(batch_next_states)
                terminal_tensor = torch.tensor(batch_terminals, dtype=torch.bool, device=self.device)
                
                # Calculate current Q values
                current_q_values = self.policy_nets[i](state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                
                # Calculate next Q values (zeroed for terminal states)
                next_q_values = torch.zeros(self.batch_size, device=self.device)
                with torch.no_grad():
                    # Only calculate for non-terminal states
                    next_q_values[~terminal_tensor] = self.target_nets[i](next_state_tensor[~terminal_tensor]).max(1)[0]
                
                # Calculate target Q values
                target_q_values = reward_tensor + self.gamma * next_q_values
                
                # Compute loss
                loss = self.loss_criterion(current_q_values, target_q_values)
                
                # Optimize the model
                self.optimizers[i].zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 1.0)
                self.optimizers[i].step()
                
            except Exception as e:
                print(f"Error in update_policy for agent {i}: {e}")
                continue
        
        # Update target networks periodically
        if self.training_steps % self.update_target_freq == 0:
            for i in range(self.num_agents):
                self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def save_model(self, filename='dqn_agent.pt'):
        """Save models to a file"""
        save_dict = {
            'epsilon': self.epsilon,
            'models': [net.state_dict() for net in self.policy_nets]
        }
        torch.save(save_dict, filename)
        
    def load_model(self, filename='dqn_agent.pt'):
        """Load models from a file"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.epsilon = checkpoint['epsilon']
        
        # Load model parameters
        for i, state_dict in enumerate(checkpoint['models']):
            self.policy_nets[i].load_state_dict(state_dict)
            self.target_nets[i].load_state_dict(state_dict)