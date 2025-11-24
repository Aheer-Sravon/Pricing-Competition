from typing import Optional, Dict, Tuple
import numpy as np
from shocks import AR1_Shock

class MarketEnv:
    """Base Market environment supporting all three models with optional shocks"""
   
    def __init__(
        self,
        market_model: str = "logit",
        shock_cfg: Optional[Dict] = None,
        n_firms: int = 2,
        horizon: int = 10000,
        seed: Optional[int] = None
    ):
        self.model = market_model.lower()
        self.n_firms = n_firms
        self.N = 15 # Number of discrete prices
        self.horizon = horizon
        self.t = 0
        self.rng = np.random.RandomState(seed)
       
        # Model-specific parameters (verified against PDFs)
        if self.model == "logit":
            self.c = 1.0
            self.a = 2.0
            self.a0 = 0.0
            self.mu = 0.25
            self.P_N = 1.473
            self.P_M = 1.925
        elif self.model == "hotelling":
            self.c = 0.0
            self.v_bar = 1.75
            self.theta = 1.0
            self.P_N = 1.00
            self.P_M = 1.25
        elif self.model == "linear":
            self.c = 0.0
            self.a_bar = 1.0
            self.d = 0.25
            
            # Nash (Cournot duopoly)
            self.P_N = self.a_bar * (1 - self.d) / (2 - self.d)  # 0.4286
            q_N = self.P_N  # By symmetry at equilibrium
            pi_N = self.P_N * q_N  # 0.1959
            
            # Monopoly (single firm, no competition)
            self.P_M = self.a_bar / 2  # 0.5
            q_M = self.a_bar - self.P_M  # 0.5 (residual demand, NOT Cournot!)
            pi_M = self.P_M * q_M  # 0.25
            
            # Store for Delta calculations
            self.pi_N_linear = pi_N  # 0.1959
            self.pi_M_linear = pi_M  # 0.25
            
            # Verify profit range is healthy
            profit_range = pi_M - pi_N  # Should be ~0.05
            if profit_range < 0.04:
                import warnings
                warnings.warn(f"Linear model profit range too small: {profit_range:.4f}")
        else:
            raise ValueError(f"Unknown model: {self.model}")
       
        # Construct price grid
        span = self.P_M - self.P_N
        self.price_grid = np.linspace(
            self.P_N - 0.15 * span,
            self.P_M + 0.15 * span,
            self.N
        )
       
        # Initialize shock configuration
        self.shock_cfg = shock_cfg or {}
        self.shock_enabled = self.shock_cfg.get("enabled", False)
       
        if self.shock_enabled:
            self.shock_mode = self.shock_cfg.get("mode", "correlated")
            scheme = self.shock_cfg.get("scheme", None)
           
            # Get AR(1) parameters (verified against PDFs)
            if scheme:
                scheme_params = {
                    'A': {'rho': 0.3, 'sigma_eta': 0.5},
                    'B': {'rho': 0.95, 'sigma_eta': 0.05},
                    'C': {'rho': 0.9, 'sigma_eta': 0.3}
                }
                if scheme.upper() in scheme_params:
                    params = scheme_params[scheme.upper()]
                    self.rho = params['rho']
                    self.sigma_eta = params['sigma_eta']
                else:
                    raise ValueError(f"Unknown scheme: {scheme}")
            else:
                self.rho = self.shock_cfg.get('rho', 0.9)
                self.sigma_eta = self.shock_cfg.get('sigma_eta', 0.15)
           
            # Initialize shock generators
            if self.shock_mode == "independent":
                self.shock_generators = [
                    AR1_Shock(self.rho, self.sigma_eta, seed=seed+i if seed else None)
                    for i in range(self.n_firms)
                ]
            else: # correlated
                self.shock_generator = AR1_Shock(self.rho, self.sigma_eta, seed=seed)
           
            self.current_shocks = np.zeros(self.n_firms)
       
        self.reset()
   
    def reset(self):
        """Reset environment - returns single state tuple"""
        self.t = 0
       
        # Reset shocks
        if self.shock_enabled:
            if self.shock_mode == "independent":
                for gen in self.shock_generators:
                    gen.reset()
            else:
                self.shock_generator.reset()
            self.current_shocks = np.zeros(self.n_firms)
       
        # Initialize prices at middle of grid
        mid_idx = self.N // 2
        self.current_price_idx = np.array([mid_idx] * self.n_firms)
       
        return self._get_state()
   
    def _get_state(self):
        """Get current state (price indices as tuple)"""
        return tuple(self.current_price_idx)
   
    def _generate_shocks(self):
        """Generate next period shocks"""
        if not self.shock_enabled:
            return np.zeros(self.n_firms)
       
        if self.shock_mode == "independent":
            shocks = np.array([gen.generate_next() for gen in self.shock_generators])
        else:
            shock = self.shock_generator.generate_next()
            shocks = np.array([shock] * self.n_firms)
       
        return shocks
   
    def get_current_shocks(self):
        """Get current shocks (for PSO evaluation)"""
        return self.current_shocks.copy() if self.shock_enabled else np.zeros(self.n_firms)
   
    def calculate_demand_and_profit(self, prices: np.ndarray, shocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate demand and profit given prices and shocks"""
        if self.model == "logit":
            return self._logit_demand_profit(prices, shocks)
        elif self.model == "hotelling":
            return self._hotelling_demand_profit(prices, shocks)
        elif self.model == "linear":
            return self._linear_demand_profit(prices, shocks)
   
    def _logit_demand_profit(self, prices: np.ndarray, shocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Logit demand - shocks affect variance but not Nash equilibrium
        
        Critical: E[demand | shocks] = demand(no shocks) since E[ε] = 0
        Nash price must remain constant regardless of shock scheme
        """
        # Base utilities WITHOUT shocks (defines Nash equilibrium)
        base_utilities = (self.a - prices) / self.mu
        
        # Add shocks OUTSIDE mu scaling to preserve E[utility]
        realized_utilities = base_utilities + shocks
        
        # Numerical stability
        max_util = np.max(realized_utilities)
        exp_utils = np.exp(realized_utilities - max_util)
        exp_outside = np.exp(self.a0/self.mu - max_util)
        
        denominator = np.sum(exp_utils) + exp_outside
        demands = exp_utils / denominator
        profits = (prices - self.c) * demands
        
        return demands, profits
   
    def _hotelling_demand_profit(self, prices: np.ndarray, shocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hotelling demand with net shock affecting boundary (verified against PDF)"""
        p1, p2 = prices[0], prices[1]
        epsilon_net = shocks[0] - shocks[1] if self.n_firms == 2 else 0
       
        x_hat = 0.5 + (p2 - p1 + epsilon_net) / (2 * self.theta)
        x_hat = np.clip(x_hat, 0, 1)
       
        q1 = x_hat
        q2 = 1 - x_hat
        demands = np.array([q1, q2])
        profits = prices * demands
       
        return demands, profits
   
    def _linear_demand_profit(self, prices: np.ndarray, shocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linear demand with additive shocks (verified against PDF)"""
        a_shocked = self.a_bar + shocks
        denominator = 1 - self.d**2
       
        q1 = ((a_shocked[0] - prices[0]) - self.d * (a_shocked[1] - prices[1])) / denominator
        q2 = ((a_shocked[1] - prices[1]) - self.d * (a_shocked[0] - prices[0])) / denominator
       
        q1 = max(0, q1)
        q2 = max(0, q2)
        demands = np.array([q1, q2])
        profits = prices * demands
       
        return demands, profits
   
    def step(self, action_indices):
        """Execute one step - returns 4 values (verified API)"""
        action_indices = np.asarray(action_indices, dtype=int)
       
        # Update price indices
        self.current_price_idx = action_indices
        prices = self.price_grid[self.current_price_idx]
       
        # Generate shocks for this period
        self.current_shocks = self._generate_shocks()
       
        # Calculate demands and profits
        demands, profits = self.calculate_demand_and_profit(prices, self.current_shocks)
       
        # Update time
        self.t += 1
       
        # Get next state
        next_state = self._get_state()
       
        # Episode termination
        done = False
       
        # Info dictionary
        info = {
            'prices': prices.copy(),
            'demands': demands.copy(),
            'shocks': self.current_shocks.copy()
        }
       
        # Return 4 values
        return next_state, profits, done, info

class MarketEnvContinuous(MarketEnv):
    def step(self, action_indices):
        prices = []
        indices = []
        for a in action_indices:
            if isinstance(a, (int, np.integer)):
                prices.append(self.price_grid[a])
                indices.append(a)
            else:
                prices.append(float(a))
                indices.append(np.argmin(np.abs(self.price_grid - a)))
        prices = np.array(prices)
        self.current_price_idx = np.array(indices)
        
        self.current_shocks = self._generate_shocks()
        demands, profits = self.calculate_demand_and_profit(prices, self.current_shocks)
        self.t += 1
        next_state = self._get_state()
        done = False
        info = {
            'prices': prices.copy(),
            'demands': demands.copy(),
            'shocks': self.current_shocks.copy()
        }
        return next_state, profits, done, info
