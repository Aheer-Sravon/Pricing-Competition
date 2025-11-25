import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.15, gamma=0.95, beta=1.5e-4, agent_id=0, price_grid=None):
        """
        Q-Learning Agent for discrete pricing decisions.
        
        Args:
            n_actions: Number of discrete price points
            alpha: Learning rate
            gamma: Discount factor
            beta: Exploration decay rate
            agent_id: Agent identifier (0 or 1)
            price_grid: Price grid for converting continuous states to indices (optional)
        """
        self.Q = np.zeros((n_actions, n_actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Exploration decay rate
        self.t = 0
        self.id = agent_id
        self.n_actions = n_actions
        self.price_grid = price_grid  # Store price grid for state conversion
    
    def _state_to_indices(self, state):
        """Convert continuous state (prices) to discrete indices
        
        Args:
            state: Can be either:
                   - tuple of indices (old format, backward compatible)
                   - numpy array of prices (new format)
                   
        Returns:
            tuple of indices
        """
        if isinstance(state, tuple):
            # Already indices (backward compatible)
            return state
        elif isinstance(state, np.ndarray):
            # Convert prices to indices
            if self.price_grid is not None:
                indices = tuple(np.argmin(np.abs(self.price_grid - price)) for price in state)
                return indices
            else:
                # Fallback: assume state is small integers that can be indices
                return tuple(state.astype(int))
        else:
            raise ValueError(f"Unknown state format: {type(state)}")

    def choose_action(self, state):
        """Choose action using epsilon-greedy with exponential decay
        
        Args:
            state: Either tuple of indices or numpy array of prices
            
        Returns:
            action: Discrete action index
        """
        state_idx = self._state_to_indices(state)
        opp_idx = state_idx[1 - self.id]
        
        epsilon = np.exp(-self.beta * self.t)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[opp_idx])  # Exploit

    def update(self, state, action, reward, next_state):
        """Update Q-table using TD learning
        
        Args:
            state: Current state (indices or prices)
            action: Action taken
            reward: Reward received
            next_state: Next state (indices or prices)
        """
        state_idx = self._state_to_indices(state)
        next_state_idx = self._state_to_indices(next_state)
        
        opp_idx = state_idx[1 - self.id]
        next_opp_idx = next_state_idx[1 - self.id]
        
        td_target = reward + self.gamma * np.max(self.Q[next_opp_idx])
        self.Q[opp_idx, action] += self.alpha * (td_target - self.Q[opp_idx, action])
        self.t += 1
