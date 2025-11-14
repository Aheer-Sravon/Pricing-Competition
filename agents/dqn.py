"""
Modern Deep Q-Network (DQN) Implementation for Algorithmic Pricing - PyTorch Version
Based on "Artificial Intelligence, Algorithmic Competition and Market Structures"
by J. Manuel Sanchez-Cartas and Evangelos Katsamakas (2022)

This implementation follows the modern DQN architecture with:
- Experience Replay Buffer
- Target Network for stability
- Epsilon-greedy exploration with decay
- 2D state representation (own price + competitor price)
- Configurable loss function (MSE vs Huber)

FRAMEWORK: PyTorch (GPU-enabled)
COMPATIBLE WITH: Your existing TensorFlow codebase (same API)
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

class ReplayBuffer:
    """Experience Replay Buffer with uniform sampling."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool):
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


class QNetwork(nn.Module):
    """Deep Q-Network using PyTorch."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_units: list = [64, 64], 
                 activation: str = 'relu', use_dueling: bool = False):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        # Select activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        
        # Build shared layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling architecture
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_units[-1], hidden_units[-1]),
                self.activation,
                nn.Linear(hidden_units[-1], 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_units[-1], hidden_units[-1]),
                self.activation,
                nn.Linear(hidden_units[-1], action_dim)
            )
        else:
            # Standard DQN
            self.output_layer = nn.Linear(hidden_units[-1], action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.shared_layers(state)
        
        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(x)
        
        return q_values


class DQNAgent:
    """
    Deep Q-Network Agent for Algorithmic Pricing - PyTorch Version.
    
    CRITICAL: State dimension MUST be 2 (not 1)
    State = (own_price_index, competitor_price_index)
    """
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int = 2,
        action_dim: int = 15,
        learning_rate: float = 0.0001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        hidden_units: list = None,
        activation: str = 'relu',
        use_double: bool = True,
        use_dueling: bool = False,
        loss_type: str = 'huber',
        gradient_clip: float = None,
        device: str = None,
        seed: int = None
    ):
        """Initialize DQN Agent (PyTorch)."""
        
        # Validate critical parameter
        assert state_dim == 2, "CRITICAL: state_dim MUST be 2!"
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store hyperparameters
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.loss_type = loss_type
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Hidden layers
        if hidden_units is None:
            hidden_units = [64, 64]
        self.hidden_units = hidden_units
        
        # Set seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Initialize networks
        self.q_network = QNetwork(
            state_dim, action_dim, hidden_units, activation, use_dueling
        ).to(self.device)
        
        self.target_network = QNetwork(
            state_dim, action_dim, hidden_units, activation, use_dueling
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Memory
        self.memory = ReplayBuffer(buffer_size)
        
        # Counters
        self.train_steps = 0
        self.episodes = 0
        
        # History
        self.loss_history = []
        self.epsilon_history = []
        
        print(f"\n{'='*70}")
        print(f"Initialized DQN Agent {agent_id} (PyTorch)")
        print(f"{'='*70}")
        print(f"State Dim: {state_dim} | Action Dim: {action_dim}")
        print(f"Hidden Units: {hidden_units}")
        print(f"Loss: {loss_type.upper()} | Double DQN: {use_double}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
    
    def get_state_representation(self, price_indices: tuple) -> np.ndarray:
        """Convert price indices to normalized 2D state."""
        state = np.array(price_indices, dtype=np.float32) / (self.action_dim - 1)
        assert state.shape == (2,), f"State must be 2D! Got {state.shape}"
        return state
    
    def select_action(self, state: tuple, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_vec = self.get_state_representation(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def remember(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def _compute_loss(self, target_q: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
        """Compute loss (MSE or Huber)."""
        if self.loss_type == 'mse':
            return F.mse_loss(current_q, target_q)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(current_q, target_q)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def replay(self) -> float:
        """Training step with experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if self.use_double:
                # Double DQN
                next_q_online = self.q_network(next_states)
                next_actions = next_q_online.argmax(dim=1)
                
                next_q_target = self.target_network(next_states)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = self._compute_loss(target_q, current_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for all actions."""
        with torch.no_grad():
            state_vec = self.get_state_representation(state)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def save(self, filepath: str):
        """Save model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes': self.episodes
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episodes = checkpoint['episodes']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    """Test the PyTorch DQN implementation."""
    
    print("="*80)
    print("DQN AGENT - PyTorch Version")
    print("="*80)
    
    # Create price grid
    n_prices = 15
    price_nash = 1.47
    price_monopoly = 2.0
    
    prices = np.linspace(
        price_nash - 0.15 * (price_monopoly - price_nash),
        price_monopoly + 0.15 * (price_monopoly - price_nash),
        n_prices
    )
    
    print(f"\nPrice Grid: [{prices[0]:.3f}, {prices[-1]:.3f}]")
    print(f"Number of Prices: {n_prices}\n")
    
    # Initialize agent
    agent = DQNAgent(
        agent_id=0,
        state_dim=2,
        action_dim=n_prices,
        loss_type='huber',
        use_double=True
    )
    
    # Test interaction
    print("Testing Basic Interaction:")
    print("-" * 80)
    
    state = (7, 7)
    print(f"State: {state}")
    
    state_vec = agent.get_state_representation(state)
    print(f"Normalized: {state_vec}")
    
    action = agent.select_action(state, explore=True)
    print(f"Action: {action} (price: ${prices[action]:.3f})")
    
    q_values = agent.get_q_values(state)
    print(f"Q-values: min={q_values.min():.4f}, max={q_values.max():.4f}")
    
    # Store experience
    agent.remember(state, action, 0.5, (action, 7), False)
    print(f"Buffer: {len(agent.memory)}/{agent.memory.buffer.maxlen}")
    
    # Network info
    print(f"\nNetwork Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    print("\n" + "="*80)
    print("PyTorch DQN Ready!")
    print("="*80)
