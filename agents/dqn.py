"""
Deep Q-Network (DQN) Implementation for Algorithmic Pricing
Based on the rlpricing-master implementation from Kastius & Schlosser (2022)
"Dynamic pricing under competition using reinforcement learning"

This implementation follows the exact architecture from the paper:
- Single input for combined state representation 
- Normalization of states by dividing by (n_actions - 1)
- Reward normalization by factor of 1000
- Double DQN with target network
- Dueling architecture
- 128 units per hidden layer
- Experience replay with uniform sampling
"""

import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Key hyperparameters from rlpricing-master
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.3
EPS_DECAY = 0.95
BATCH_SIZE = 32
NORM_FACTOR = 1000  # Reward normalization factor
TARGET_UPDATE_FREQ = 256  # Update target network every N samples
BUFFER_SIZE = 10000
LEARNING_RATE = 0.0005


class ReplayBuffer:
    """Experience Replay Buffer with uniform sampling (matching rlpricing-master)."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([exp.done for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture matching rlpricing-master.
    Uses separate value and advantage streams.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingQNetwork, self).__init__()
        
        # Shared layers (128 units each, matching rlpricing-master)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream  
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Initialize weights using Xavier (glorot_normal in Keras)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared(x)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage (subtracting mean for stability)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class DQNAgent:
    """
    DQN Agent for Algorithmic Pricing matching rlpricing-master implementation.
    
    Key features:
    - State normalization by (n_actions - 1)
    - Reward normalization by NORM_FACTOR
    - Double DQN with target network
    - Dueling architecture
    - Epsilon decay schedule matching paper
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int = 2,  # Must be 2 for pricing (own_price, competitor_price)
        action_dim: int = 15,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_start: float = EPS_START,
        epsilon_end: float = EPS_END,
        epsilon_decay: float = EPS_DECAY,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = BUFFER_SIZE,
        norm_factor: float = NORM_FACTOR,
        device: str = None,
        seed: int = None
    ):
        """Initialize DQN Agent matching rlpricing-master."""
        
        assert state_dim == 2, "State dimension must be 2 for pricing problems"
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store parameters
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.norm_factor = norm_factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Set seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Initialize networks
        self.q_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer (Adam with specific learning rate)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training counters
        self.train_steps = 0
        self.episodes = 0
        self.samples_processed = 0
        
        # History tracking
        self.loss_history = []
        self.epsilon_history = []
        
        print(f"\n{'='*70}")
        print(f"Initialized DQN Agent {agent_id} (rlpricing-master version)")
        print(f"{'='*70}")
        print(f"State Dim: {state_dim} | Action Dim: {action_dim}")
        print(f"Architecture: Dueling DQN with Double Q-learning")
        print(f"Hidden Units: [128, 128, 128] + Dueling [128]")
        print(f"Norm Factor: {norm_factor} | Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
    
    def get_state_representation(self, price_indices: tuple) -> np.ndarray:
        """
        Convert price indices to normalized state representation.
        CRITICAL: Normalize by (action_dim - 1) as in rlpricing-master
        """
        # Convert to numpy array and normalize
        state = np.array(price_indices, dtype=np.float32) / (self.action_dim - 1)
        assert state.shape == (2,), f"State must be 2D, got shape {state.shape}"
        return state
    
    def select_action(self, state: tuple, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        State is tuple of (own_price_idx, competitor_price_idx)
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            # Get normalized state representation
            state_vec = self.get_state_representation(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def remember(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool):
        """
        Store experience in replay buffer.
        States are stored as normalized vectors.
        """
        # Convert states to normalized representations
        state_vec = self.get_state_representation(state)
        next_state_vec = self.get_state_representation(next_state)
        
        # Store normalized states
        self.memory.push(state_vec, action, reward, next_state_vec, done)
    
    def replay(self) -> float:
        """
        Training step with experience replay.
        Matches the training procedure from rlpricing-master.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Normalize rewards (critical for matching paper behavior)
        rewards = rewards / self.norm_factor
        
        # Current Q-values for chosen actions
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            # Online network selects best action
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)
            
            # Target network evaluates the selected action
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss (MSE as in rlpricing-master)
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update counters
        self.samples_processed += self.batch_size
        self.train_steps += 1
        
        # Update target network periodically
        if self.samples_processed % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self):
        """
        Update exploration rate with multiplicative decay.
        Matches rlpricing-master epsilon schedule.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_vec = self.get_state_representation(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'samples_processed': self.samples_processed
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episodes = checkpoint['episodes']
        self.samples_processed = checkpoint.get('samples_processed', 0)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    """Test the DQN implementation."""
    
    print("="*80)
    print("DQN AGENT - rlpricing-master Version")
    print("="*80)
    
    # Create price grid (matching paper)
    n_prices = 15
    price_nash = 1.47
    price_monopoly = 2.0
    
    prices = np.linspace(
        price_nash - 0.15 * (price_monopoly - price_nash),
        price_monopoly + 0.15 * (price_monopoly - price_nash),
        n_prices
    )
    
    print(f"\nPrice Grid: [{prices[0]:.3f}, {prices[-1]:.3f}]")
    print(f"Number of Prices: {n_prices}")
    print(f"Nash Price: {price_nash:.3f}")
    print(f"Monopoly Price: {price_monopoly:.3f}\n")
    
    # Initialize agent
    agent = DQNAgent(
        agent_id=0,
        state_dim=2,
        action_dim=n_prices,
        seed=42
    )
    
    # Test basic functionality
    print("Testing Agent Functionality:")
    print("-" * 80)
    
    # Test state representation
    state = (7, 7)  # Middle of price grid
    print(f"State (indices): {state}")
    
    state_vec = agent.get_state_representation(state)
    print(f"Normalized state: {state_vec}")
    print(f"Expected: [{7/(n_prices-1):.3f}, {7/(n_prices-1):.3f}]")
    
    # Test action selection
    action = agent.select_action(state, explore=False)
    print(f"\nSelected action (no exploration): {action}")
    print(f"Corresponding price: ${prices[action]:.3f}")
    
    # Test Q-values
    q_values = agent.get_q_values(state)
    print(f"\nQ-values shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    
    # Test memory and training
    print("\nTesting Memory and Training:")
    print("-" * 80)
    
    # Add some experiences
    for i in range(50):
        s = (random.randint(0, n_prices-1), random.randint(0, n_prices-1))
        a = random.randint(0, n_prices-1)
        r = random.random() * 100  # Random reward
        s_next = (random.randint(0, n_prices-1), random.randint(0, n_prices-1))
        agent.remember(s, a, r, s_next, False)
    
    print(f"Memory size: {len(agent.memory)}/{agent.memory.buffer.maxlen}")
    
    # Test training
    loss = agent.replay()
    print(f"Training loss: {loss:.6f}")
    
    # Test epsilon decay
    initial_eps = agent.epsilon
    agent.update_epsilon()
    print(f"\nEpsilon decay: {initial_eps:.4f} -> {agent.epsilon:.4f}")
    
    # Network info
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    print(f"\nNetwork Parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print("DQN Agent (rlpricing-master version) Ready!")
    print("="*80)
