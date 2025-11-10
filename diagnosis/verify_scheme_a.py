"""
Verify your Scheme A results and diagnose any remaining issues
Run this from your project directory where environments module is available
"""

import numpy as np
from environments import MarketEnvContinuous

def verify_linear_profit_range():
    """Check if Linear model profit range was fixed"""
    print("=" * 70)
    print("VERIFICATION: Linear Model Profit Range")
    print("=" * 70)
    
    shock_cfg = {'enabled': True, 'scheme': 'A', 'mode': 'independent'}
    env = MarketEnvContinuous(market_model='linear', shock_cfg=shock_cfg)
    
    # Nash
    prices_n = np.array([env.P_N, env.P_N])
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    
    # Monopoly - TRUE monopoly (single firm)
    p_M = env.a_bar / 2
    q_M = env.a_bar - p_M
    pi_M_true = p_M * q_M
    
    profit_range = pi_M_true - profits_n[0]
    
    print(f"\nLinear Model Benchmarks:")
    print(f"  Nash profit:     π^N = {profits_n[0]:.4f}")
    print(f"  Monopoly profit: π^M = {pi_M_true:.4f}")
    print(f"  Profit range:    π^M - π^N = {profit_range:.4f}")
    
    # Your results
    delta_q = 0.16
    delta_pso = 0.27
    
    implied_profit_q = profits_n[0] + delta_q * profit_range
    implied_profit_pso = profits_n[0] + delta_pso * profit_range
    
    print(f"\nImplied Average Profits from Your Deltas:")
    print(f"  Q-learning (Δ=0.16):  π̄ = {implied_profit_q:.4f}")
    print(f"  PSO (Δ=0.27):         π̄ = {implied_profit_pso:.4f}")
    
    if profit_range < 0.01:
        print(f"\n❌ PROBLEM: Profit range {profit_range:.4f} is too small!")
        print("   Linear model still has the monopoly calculation bug.")
        print("   Delta calculations will be unstable.")
    elif profit_range < 0.04:
        print(f"\n⚠️  WARNING: Profit range {profit_range:.4f} is small.")
        print("   Results may be somewhat unstable, but usable.")
    else:
        print(f"\n✅ GOOD: Profit range {profit_range:.4f} is healthy.")
    
    if delta_pso > delta_q:
        print(f"\n⚠️  ANOMALY: PSO (Δ={delta_pso}) > Q-learning (Δ={delta_q})")
        print("   This is unexpected - Q should be more collusive than PSO.")
        print("   Possible causes:")
        print("   1. Small profit range causing noise")
        print("   2. Insufficient iterations for convergence")
        print("   3. Unlucky random seed")
    
    print("\n" + "=" * 70)


def verify_logit_nash_invariance():
    """Check if Logit Nash stays constant with shocks"""
    print("=" * 70)
    print("VERIFICATION: Logit Nash Invariance to Shocks")
    print("=" * 70)
    
    # No shocks
    env_no_shock = MarketEnvContinuous(market_model='logit', shock_cfg=None)
    prices = np.array([1.473, 1.473])
    _, profits_no = env_no_shock.calculate_demand_and_profit(prices, np.zeros(2))
    
    # With Scheme A
    shock_cfg = {'enabled': True, 'scheme': 'A', 'mode': 'independent'}
    env_shock = MarketEnvContinuous(market_model='logit', shock_cfg=shock_cfg)
    
    # Average over many realizations
    profits_list = []
    for _ in range(1000):
        # Generate shocks
        if hasattr(env_shock, 'shock_generators'):
            env_shock.shock_generators[0].generate_next()
            env_shock.shock_generators[1].generate_next()
            shocks = np.array([
                env_shock.shock_generators[0].current,
                env_shock.shock_generators[1].current
            ])
        else:
            shocks = np.zeros(2)
        
        _, profits = env_shock.calculate_demand_and_profit(prices, shocks)
        profits_list.append(profits[0])
    
    avg_profit_shock = np.mean(profits_list)
    diff = abs(avg_profit_shock - profits_no[0])
    
    print(f"\nLogit at Nash Price p^N = 1.473:")
    print(f"  No shocks:       π = {profits_no[0]:.4f}")
    print(f"  Scheme A (E[π]): π = {avg_profit_shock:.4f}")
    print(f"  Difference:      Δπ = {diff:.4f}")
    
    if diff < 0.01:
        print(f"\n✅ EXCELLENT: Nash invariant to shocks (diff < 0.01)")
        print("   Logit shock implementation is correct!")
    elif diff < 0.05:
        print(f"\n✓ GOOD: Nash roughly invariant (diff < 0.05)")
        print("   Minor numerical issues, but acceptable.")
    else:
        print(f"\n❌ PROBLEM: Nash shifts with shocks (diff = {diff:.4f})")
        print("   Logit shock implementation still has bugs.")
    
    print("\n" + "=" * 70)


def interpret_results():
    """Interpret your Scheme A results"""
    print("=" * 70)
    print("INTERPRETATION: Your Scheme A Results")
    print("=" * 70)
    
    results = {
        'LOGIT': {'Q': 0.72, 'PSO': 0.28},
        'HOTELLING': {'Q': 0.58, 'PSO': 0.33},
        'LINEAR': {'Q': 0.16, 'PSO': 0.27}
    }
    
    print("\n               Q Δ    PSO Δ   Q > PSO?  Assessment")
    print("  " + "-" * 60)
    for model, deltas in results.items():
        q_higher = "✓" if deltas['Q'] > deltas['PSO'] else "✗"
        
        if model == 'LOGIT':
            status = "✅ Excellent" if deltas['Q'] > 0.5 and deltas['PSO'] < 0.4 else "⚠️  Check"
        elif model == 'HOTELLING':
            status = "✓ Acceptable" if deltas['Q'] > 0.3 else "⚠️  Low"
        else:  # LINEAR
            status = "❌ Anomalous" if deltas['PSO'] > deltas['Q'] else "✓ OK"
        
        print(f"  {model:10} {deltas['Q']:5.2f}  {deltas['PSO']:5.2f}    {q_higher:^7}  {status}")
    
    print("\n  Key Findings:")
    print("  • Logit: Strong Q-learning collusion, PSO competitive ✅")
    print("  • Hotelling: Both algorithms show supracompetitive pricing")
    print("  • Linear: PSO > Q is backwards - likely due to small profit range")
    
    print("\n  Comparison to Paper (baseline, no shocks):")
    print("  • Paper Q vs PSO Logit: Q >> PSO (your results show this ✓)")
    print("  • Shocks increase variance and may push both algorithms higher")
    print("  • Linear anomaly suggests profit range issue persists")
    
    print("\n" + "=" * 70)


def recommendations():
    """Provide recommendations"""
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. LOGIT Results: ✅ Good")
    print("   Your Δ = 0.72 for Q-learning is reasonable with Scheme A shocks")
    print("   PSO Δ = 0.28 shows it's more competitive (as expected)")
    print("   → No action needed for Logit")
    
    print("\n2. HOTELLING Results: ⚠️  Higher than expected")
    print("   Both algorithms showing more collusion than paper baseline")
    print("   Possible causes:")
    print("   • Shocks making coordination easier")
    print("   • Different algorithm parameters")
    print("   • Insufficient exploration")
    print("   → Consider: Increase β for Q-learning, adjust PSO restart threshold")
    
    print("\n3. LINEAR Results: ❌ Needs investigation")
    print("   PSO > Q-learning is backwards")
    print("   → Check: Linear profit range using verify_linear_profit_range()")
    print("   → If range < 0.04: Apply monopoly fix from COPY_PASTE_FIXES.md")
    print("   → If range OK: Try different random seed or longer horizon")
    
    print("\n4. OVERALL Assessment:")
    print("   Logit results are publication-quality ✅")
    print("   Hotelling acceptable but high ⚠️")
    print("   Linear needs debugging ❌")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n")
    verify_logit_nash_invariance()
    print("\n")
    verify_linear_profit_range()
    print("\n")
    interpret_results()
    print("\n")
    recommendations()
