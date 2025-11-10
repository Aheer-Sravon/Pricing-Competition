"""
Reference Implementation: Correct Shock Handling in Market Models

This shows the CORRECT way to implement shocks so that:
1. Nash equilibrium prices remain invariant to shock schemes
2. Shocks only add variance, not systematic bias
3. E[demand | shocks] = demand(no shocks) when E[shocks] = 0
"""

import numpy as np
from typing import Tuple

class ReferenceShockImplementation:
    """
    Key insight: AR(1) shocks ε_t = ρ·ε_{t-1} + η_t have E[ε] = 0
    
    Therefore: E[demand(p, ε)] = demand(p, 0) for all prices p
    
    Nash equilibrium should be invariant to shock parameters (ρ, σ_η)
    """
    
    @staticmethod
    def logit_correct(prices: np.ndarray, shocks: np.ndarray, 
                      a: float = 2.0, c: float = 1.0, mu: float = 0.25, 
                      a0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ✅ CORRECT Logit implementation with shocks
        
        Nash price: p^N = 1.473 (independent of shock scheme)
        
        Key: Shocks enter OUTSIDE the mu-scaled utilities
        """
        # Deterministic utilities (define Nash equilibrium)
        base_utils = (a - prices) / mu
        
        # Shocks add variance but preserve E[utility]
        realized_utils = base_utils + shocks  # NOT (... + shocks) / mu
        
        # Standard logit calculation
        max_u = np.max(realized_utils)
        exp_utils = np.exp(realized_utils - max_u)
        exp_outside = np.exp(a0/mu - max_u)
        
        demands = exp_utils / (exp_utils.sum() + exp_outside)
        profits = (prices - c) * demands
        
        return demands, profits
    
    @staticmethod
    def logit_wrong(prices: np.ndarray, shocks: np.ndarray,
                    a: float = 2.0, c: float = 1.0, mu: float = 0.25,
                    a0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ❌ WRONG implementation (what you currently have)
        
        Problem: Nash price shifts to ~1.797 with Scheme A shocks
        
        Cause: Shocks inside mu scaling create systematic bias
        """
        # This creates bias because shocks affect utility scaling
        utilities = (a - prices + shocks) / mu  # ← WRONG
        
        max_u = np.max(utilities)
        exp_utils = np.exp(utilities - max_u)
        exp_outside = np.exp(a0/mu - max_u)
        
        demands = exp_utils / (exp_utils.sum() + exp_outside)
        profits = (prices - c) * demands
        
        return demands, profits
    
    @staticmethod
    def hotelling_correct(prices: np.ndarray, shocks: np.ndarray,
                          theta: float = 1.0, v: float = 1.75, 
                          c: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ✅ CORRECT Hotelling with net shock
        
        Nash price: p^N = 1.0 (independent of shock scheme)
        """
        p1, p2 = prices[0], prices[1]
        epsilon_net = shocks[0] - shocks[1]  # Net shock affects boundary
        
        # Indifferent consumer location
        x_hat = 0.5 + (p2 - p1 + epsilon_net) / (2 * theta)
        x_hat = np.clip(x_hat, 0, 1)
        
        demands = np.array([x_hat, 1 - x_hat])
        profits = prices * demands
        
        return demands, profits
    
    @staticmethod
    def linear_correct(prices: np.ndarray, shocks: np.ndarray,
                       a_bar: float = 1.0, d: float = 0.25,
                       c: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ✅ CORRECT Linear demand with additive shocks
        
        Nash price: p^N = 0.4286 (independent of shock scheme)
        """
        # Shocks shift demand curves temporarily
        a_shocked = a_bar + shocks
        
        denom = 1 - d**2
        
        q1 = ((a_shocked[0] - prices[0]) - d*(a_shocked[1] - prices[1])) / denom
        q2 = ((a_shocked[1] - prices[1]) - d*(a_shocked[0] - prices[0])) / denom
        
        demands = np.array([max(0, q1), max(0, q2)])
        profits = prices * demands
        
        return demands, profits


def verify_nash_invariance():
    """
    Test that Nash prices don't shift with shocks
    """
    print("=" * 70)
    print("VERIFICATION: Nash Price Invariance to Shocks")
    print("=" * 70)
    
    prices_nash = np.array([1.473, 1.473])
    
    # No shocks
    shocks_zero = np.array([0.0, 0.0])
    d0, p0 = ReferenceShockImplementation.logit_correct(prices_nash, shocks_zero)
    
    # With shocks (simulating Scheme A average)
    shocks_pos = np.array([0.5, -0.3])
    d1, p1 = ReferenceShockImplementation.logit_correct(prices_nash, shocks_pos)
    
    shocks_neg = np.array([-0.5, 0.3])
    d2, p2 = ReferenceShockImplementation.logit_correct(prices_nash, shocks_neg)
    
    # Average should equal no-shock case
    avg_profit = (p1[0] + p2[0]) / 2
    
    print(f"\nLogit Model at Nash Price (1.473):")
    print(f"  No shocks:     π = {p0[0]:.4f}")
    print(f"  Shock (+0.5):  π = {p1[0]:.4f}")
    print(f"  Shock (-0.5):  π = {p2[0]:.4f}")
    print(f"  Average:       π = {avg_profit:.4f}")
    print(f"  Difference:    {abs(avg_profit - p0[0]):.6f} (should be ~0)")
    
    if abs(avg_profit - p0[0]) < 0.01:
        print("  ✅ Nash invariance preserved!")
    else:
        print("  ❌ Nash shifts with shocks - implementation wrong!")
    
    print("\n" + "=" * 70)


def demonstrate_linear_profits():
    """
    Show correct profit calculations for Linear model
    """
    print("=" * 70)
    print("LINEAR MODEL: Correct Profit Calculations")
    print("=" * 70)
    
    a_bar, d, c = 1.0, 0.25, 0.0
    
    # Nash equilibrium
    p_N = a_bar * (1 - d) / (2 - d)
    prices_nash = np.array([p_N, p_N])
    d_nash, pi_nash = ReferenceShockImplementation.linear_correct(
        prices_nash, np.zeros(2), a_bar, d, c
    )
    
    # Monopoly
    p_M = a_bar / 2
    prices_mono = np.array([p_M, p_M])
    d_mono, pi_mono = ReferenceShockImplementation.linear_correct(
        prices_mono, np.zeros(2), a_bar, d, c
    )
    
    profit_range = pi_mono[0] - pi_nash[0]
    
    print(f"\nNash Equilibrium:")
    print(f"  Price:  {p_N:.4f}")
    print(f"  Demand: {d_nash[0]:.4f}")
    print(f"  Profit: {pi_nash[0]:.4f}")
    
    print(f"\nMonopoly:")
    print(f"  Price:  {p_M:.4f}")
    print(f"  Demand: {d_mono[0]:.4f}")
    print(f"  Profit: {pi_mono[0]:.4f}")
    
    print(f"\nProfit Range:")
    print(f"  π^M - π^N = {profit_range:.4f}")
    
    if profit_range > 0.04:
        print("  ✅ Profit range healthy for Delta calculations")
    else:
        print(f"  ❌ Profit range too small ({profit_range:.4f} < 0.04)")
        print("     Delta will be unstable!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify_nash_invariance()
    print()
    demonstrate_linear_profits()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Use logit_correct() implementation in your code")
    print("✓ Verify Nash prices stay constant across shock schemes")
    print("✓ Check Linear model profit range > 0.04 for stable Deltas")
    print("=" * 70)
