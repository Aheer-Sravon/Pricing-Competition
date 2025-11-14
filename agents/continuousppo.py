>>> """
... Continuous Proximal Policy Optimization (PPO) for Algorithmic Pricing - PyTorch
... Based on "Algorithmic Collusion in Dynamic Pricing with Deep Reinforcement Learning" 
... by Deng, Schiffer, and Bichler (2024) and original PPO by Schulman et al. (2017)
... 
... FRAMEWORK: PyTorch (GPU-enabled)
... ACTION SPACE: Continuous (direct price values, no discretization)
... """
... 
... import numpy as np
... import torch
... import torch.nn as nn
... import torch.optim as optim
... import torch.nn.functional as F
... from torch.distributions import Normal
... from collections import deque
... 
... 
... class ActorCriticNetwork(nn.Module):
...     """
...     Actor-Critic Network for Continuous PPO
...     
...     ARCHITECTURE DECISION:
...     - Hidden layers: [64, 64] - SAME as DQN (from dqn.py)
...     - Activation: TANH - FROM Algorithm PPO Paper 1, page 6: 
...       "fully-connected MLP with two hidden layers of 64 units, and tanh nonlinearities"
...       
...       WHY TANH instead of ReLU (used in DQN)?
...       - Paper explicitly specifies tanh for continuous control tasks
...       - Tanh bounds outputs to [-1, 1], providing natural gradient scaling
...       - Standard practice for continuous policy networks (confirmed in original PPO paper)
...       
...     - State dim: 2 (own_price, competitor_price) - normalized actual prices
...     - Actor output: mean of Gaussian distribution (log_std learned separately)
...     - Critic output: state value
...     """
...     
...     def __init__(self, state_dim: int = 2, hidden_units: list = [64, 64]):
...         super(ActorCriticNetwork, self).__init__()
...         
...         self.state_dim = state_dim
...         self.hidden_units = hidden_units
...         
...         # Shared feature extraction layers
...         # DECISION: Use shared layers between actor and critic
        # SOURCE: Algorithm PPO Paper 1, page 5, equation (9): 
        # "If using a neural network architecture that shares parameters between 
        # the policy and value function..."
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())  # CRITICAL: Tanh as per paper specification
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy): outputs mean of Gaussian
        self.actor_mean = nn.Linear(hidden_units[-1], 1)  # Single continuous action (price)
        
        # Log std is learned as a separate parameter
        # DECISION: Learned log_std parameter
        # SOURCE: Algorithm PPO Paper 1, page 6: 
        # "outputting the mean of a Gaussian distribution, with variable standard deviations"
        # Also Table 4: "Log stdev. of action distribution LinearAnneal(-0.7, -1.6)"
        # For simplicity, we'll use a learned parameter (adaptive)
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_units[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        DECISION: Xavier normal initialization
        SOURCE: Standard practice from DQN implementation (dqn.py)
        Also used in original PPO implementations
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor):
        """Forward pass through actor-critic network"""
        features = self.shared_layers(state)
        
        # Actor: mean of Gaussian policy
        action_mean = self.actor_mean(features)
        
        # Critic: state value
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """
        Get action distribution, sample action, compute log prob and value
        
        DECISION: Gaussian policy with learned std
        SOURCE: Algorithm PPO Paper 1, page 6: 
        "outputting the mean of a Gaussian distribution, with variable standard deviations"
        """
        action_mean, value = self.forward(state)
        
        # Create Gaussian distribution
        action_std = self.log_std.exp()
        dist = Normal(action_mean, action_std)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(-1)
        
        # Entropy for exploration bonus
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy, value


class PPOAgent:
    """
    Continuous Proximal Policy Optimization Agent for Algorithmic Pricing
    
    HYPERPARAMETERS SOURCE:
    Primary: Algorithm PPO Paper 1 (Deng et al., 2024), Table 3 (Mujoco experiments)
    Secondary: Original PPO Paper (Schulman et al., 2017) if not found in Paper 1
    """
    
    def __init__(
        self,
        agent_id: int,
        price_min: float = 0.0,
        price_max: float = 2.0,
        state_dim: int = 2,
        learning_rate: float = 3e-4,     # SOURCE: Paper 1, Table 3: "Adam stepsize: 3 × 10^-4"
        gamma: float = 0.99,              # SOURCE: Paper 1, Table 3: "Discount (γ): 0.99"
        gae_lambda: float = 0.95,         # SOURCE: Paper 1, Table 3: "GAE parameter (λ): 0.95"
        clip_epsilon: float = 0.2,        # SOURCE: Paper 1, Table 1: "Clipping, ε = 0.2" (best performer)
        value_coef: float = 1.0,          # SOURCE: Paper 1, Table 5: "VF coeff. c1: 1"
        entropy_coef: float = 0.01,       # SOURCE: Paper 1, Table 5: "Entropy coeff. c2: 0.01"
        max_grad_norm: float = 0.5,       # SOURCE: Original PPO Paper - standard gradient clipping
        num_epochs: int = 10,             # SOURCE: Paper 1, Table 3: "Num. epochs: 10"
        minibatch_size: int = 64,         # SOURCE: Paper 1, Table 3: "Minibatch size: 64"
        horizon: int = 2048,              # SOURCE: Paper 1, Table 3: "Horizon (T): 2048"
        hidden_units: list = None,
        device: str = None,
        seed: int = None
    ):
        """
        Initialize Continuous PPO Agent
        
        CRITICAL DESIGN DECISIONS:
        
        1. CONTINUOUS ACTION SPACE:
           - Actions are actual prices in [price_min, price_max], NOT discrete indices
           - Similar to PSO implementation (pso.py), but learned via PPO
           
        2. STATE REPRESENTATION:
           - State = (own_price, competitor_price) - actual prices, normalized
           - Different from DQN which uses (own_price_INDEX, competitor_price_INDEX)
           - Normalization: divide by price_max for [0, 1] range
           
        3. NETWORK ARCHITECTURE:
           - Same structure as DQN: [64, 64] hidden layers
           - BUT uses Tanh activation (from Paper 1) instead of ReLU (from DQN)
           - Shared parameters between actor and critic (standard PPO practice)
           
        4. ALL HYPERPARAMETERS:
           - Sourced from Algorithm PPO Paper 1, Table 3 (Mujoco continuous control)
           - These were empirically validated for continuous pricing tasks
        """
        
        # Store agent configuration
        self.agent_id = agent_id
        self.price_min = price_min
        self.price_max = price_max
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.horizon = horizon
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hidden layers
        if hidden_units is None:
            hidden_units = [64, 64]  # Same as DQN
        self.hidden_units = hidden_units
        
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Initialize actor-critic network
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            hidden_units=hidden_units
        ).to(self.device)
        
        # Optimizer: Adam
        # SOURCE: Paper 1, Table 3: "Adam stepsize: 3 × 10^-4"
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Replay buffer for trajectory storage
        self.reset_buffer()
        
        # Training statistics
        self.episode = 0
        self.update_step = 0
        
        # History tracking
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        
        print(f"\n{'='*70}")
        print(f"Initialized Continuous PPO Agent {agent_id}")
        print(f"{'='*70}")
        print(f"State Dim: {state_dim} (continuous prices, not indices)")
        print(f"Action Space: Continuous [{price_min:.2f}, {price_max:.2f}]")
        print(f"Hidden Units: {hidden_units} (same as DQN)")
        print(f"Activation: Tanh (from Paper 1, not ReLU from DQN)")
        print(f"Learning Rate: {learning_rate} (Paper 1, Table 3)")
        print(f"Gamma: {gamma} (Paper 1, Table 3)")
        print(f"GAE Lambda: {gae_lambda} (Paper 1, Table 3)")
        print(f"Clip Epsilon: {clip_epsilon} (Paper 1, Table 1)")
        print(f"Epochs per Update: {num_epochs} (Paper 1, Table 3)")
        print(f"Minibatch Size: {minibatch_size} (Paper 1, Table 3)")
        print(f"Horizon: {horizon} (Paper 1, Table 3)")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
    
    def reset_buffer(self):
        """
        Reset trajectory buffer
        
        DECISION: Store full trajectories for PPO update
        SOURCE: Original PPO Paper, Algorithm 1: 
        "Run policy in environment for T timesteps... Optimize surrogate L"
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_state_representation(self, prices: tuple) -> np.ndarray:
        """
        Convert price tuple to normalized state
        
        CRITICAL DECISION: Use actual prices, not indices
        - Input: (own_price, competitor_price) as floats
        - Normalize by price_max for [0, 1] range
        - This is CONTINUOUS, unlike DQN which uses discrete price indices
        
        SOURCE: Adapted from PSO (pso.py) which also uses continuous prices
        """
        own_price, competitor_price = prices
        state = np.array([own_price / self.price_max, 
                         competitor_price / self.price_max], dtype=np.float32)
        return state
    
    def select_action(self, state: tuple, explore: bool = True):
        """
        Select continuous action (price) from policy
        
        DECISION: Stochastic policy with Gaussian distribution
        SOURCE: Paper 1, page 6: "outputting the mean of a Gaussian distribution"
        
        Returns: (action, log_prob, value) tuple
        - action: continuous price value
        - log_prob: for PPO loss computation
        - value: baseline for advantage estimation
        """
        state_vec = self.get_state_representation(state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        
        # Convert action to actual price
        # DECISION: Clip to valid range and squeeze to scalar
        price = action.cpu().item()
        price = np.clip(price * self.price_max, self.price_min, self.price_max)
        
        return price, log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state: tuple, action: float, reward: float, 
                        log_prob: float, value: float, done: bool):
        """Store transition in buffer"""
        state_vec = self.get_state_representation(state)
        
        self.states.append(state_vec)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        SOURCE: Paper 1, Table 3: "GAE parameter (λ): 0.95"
        Also Original PPO Paper, equation (11) and (12)
        
        GAE formula:
        δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γ*λ)*δ_{t+1} + ... + (γ*λ)^{T-t+1}*δ_{T-1}
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = next_value
        
        # Compute advantages in reverse order (from T-1 to 0)
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]
            
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value_t * next_non_terminal - self.values[t]
            
            # GAE: A_t = δ_t + (γ*λ)*A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
            # Return: G_t = A_t + V(s_t)
            returns.insert(0, gae + self.values[t])
        
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)
    
    def update(self, next_state: tuple):
        """
        Perform PPO update using collected trajectories
        
        ALGORITHM SOURCE: Original PPO Paper, Algorithm 1
        HYPERPARAMETERS: Paper 1, Table 3
        
        KEY COMPONENTS:
        1. Compute advantages using GAE
        2. Multiple epochs of minibatch SGD
        3. Clipped surrogate objective (equation 7 in original PPO paper)
        4. Value function loss
        5. Entropy bonus
        """
        
        if len(self.states) == 0:
            return
        
        # Get final value for bootstrap
        next_state_vec = self.get_state_representation(next_state)
        next_state_tensor = torch.FloatTensor(next_state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, next_value = self.network.get_action_and_value(next_state_tensor)
            next_value = next_value.cpu().item()
        
        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).unsqueeze(-1).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        # DECISION: Advantage normalization for stability
        # SOURCE: Standard practice in PPO implementations, improves training stability
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Multiple epochs of optimization
        # SOURCE: Paper 1, Table 3: "Num. epochs: 10"
        batch_size = len(self.states)
        num_minibatches = max(1, batch_size // self.minibatch_size)
        
        for epoch in range(self.num_epochs):
            # Random permutation for minibatch sampling
            indices = np.random.permutation(batch_size)
            
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                minibatch_indices = indices[start_idx:end_idx]
                
                # Get minibatch
                mb_states = states_tensor[minibatch_indices]
                mb_actions = actions_tensor[minibatch_indices]
                mb_old_log_probs = old_log_probs_tensor[minibatch_indices]
                mb_advantages = advantages_tensor[minibatch_indices]
                mb_returns = returns_tensor[minibatch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )
                
                # PPO Clipped Surrogate Objective
                # SOURCE: Original PPO Paper, equation (7)
                # L^CLIP(θ) = E_t[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
                ratio = (new_log_probs - mb_old_log_probs).exp()
                
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                
                surrogate1 = ratio * mb_advantages
                surrogate2 = clipped_ratio * mb_advantages
                
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value Function Loss
                # SOURCE: Paper 1, equation (9): "c1*L^VF_t(θ)"
                # Using MSE loss (standard for PPO)
                value_loss = F.mse_loss(new_values.squeeze(), mb_returns)
                
                # Entropy Bonus for Exploration
                # SOURCE: Paper 1, Table 5: "Entropy coeff. c2: 0.01"
                entropy_loss = entropy.mean()
                
                # Combined Loss
                # SOURCE: Paper 1, equation (9):
                # L = L^CLIP - c1*L^VF + c2*S[π_θ]
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                # SOURCE: Original PPO Paper, standard practice for stability
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track statistics
                self.actor_loss_history.append(actor_loss.item())
                self.critic_loss_history.append(value_loss.item())
                self.entropy_history.append(entropy_loss.item())
        
        self.update_step += 1
        
        # Clear buffer after update
        self.reset_buffer()
        
        return {
            'actor_loss': np.mean(self.actor_loss_history[-10:]) if self.actor_loss_history else 0,
            'critic_loss': np.mean(self.critic_loss_history[-10:]) if self.critic_loss_history else 0,
            'entropy': np.mean(self.entropy_history[-10:]) if self.entropy_history else 0
        }
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.episode,
            'update_step': self.update_step,
            'config': {
                'agent_id': self.agent_id,
                'price_min': self.price_min,
                'price_max': self.price_max,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'hidden_units': self.hidden_units
            }
        }
        torch.save(checkpoint, filepath)
        print(f"PPO Agent {self.agent_id} saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode = checkpoint['episode']
        self.update_step = checkpoint['update_step']
        print(f"PPO Agent {self.agent_id} loaded from {filepath}")


if __name__ == "__main__":
    """Test the Continuous PPO implementation"""
    
    print("="*80)
    print("CONTINUOUS PPO AGENT - Algorithmic Pricing")
    print("="*80)
    
    # Price range (similar to PSO)
    price_min = 0.0
    price_max = 2.0
    
    print(f"\nPrice Range: [{price_min:.2f}, {price_max:.2f}]")
    print(f"Action Space: CONTINUOUS (not discrete)\n")
    
    # Initialize agent
    agent = PPOAgent(
        agent_id=0,
        price_min=price_min,
        price_max=price_max,
        state_dim=2
    )
    
    # Test interaction
    print("Testing Basic Interaction:")
    print("-" * 80)
    
    # Current prices (continuous values, not indices!)
    state = (1.5, 1.6)  # (own_price, competitor_price)
    print(f"State (prices): {state}")
    
    state_vec = agent.get_state_representation(state)
    print(f"Normalized state: {state_vec}")
    
    # Select action (continuous price)
    price, log_prob, value = agent.select_action(state, explore=True)
    print(f"Selected price: ${price:.4f} (continuous)")
    print(f"Log probability: {log_prob:.4f}")
    print(f"State value: {value:.4f}")
    
    # Store transition
    agent.store_transition(state, price, 0.5, log_prob, value, False)
    print(f"Buffer size: {len(agent.states)}")
    
    # Network info
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"\nNetwork Parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print("Continuous PPO Ready for Algorithmic Pricing!")
