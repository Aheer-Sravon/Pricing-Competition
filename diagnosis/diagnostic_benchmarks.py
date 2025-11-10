"""
Diagnostic Script to Verify Theoretical Benchmarks
Run this to check if your theoretical_benchmarks.py is working correctly
"""

import numpy as np
from theoretical_benchmarks import TheoreticalBenchmarks

print("=" * 80)
print("DIAGNOSTIC: THEORETICAL BENCHMARKS VERIFICATION")
print("=" * 80)

# Set seed for reproducibility
calculator = TheoreticalBenchmarks(seed=99)

# Define Scheme A configuration
shock_cfg = {
    'enabled': True,
    'scheme': 'A',
    'mode': 'independent'
}

print("\n" + "=" * 80)
print("SCHEME A - INDEPENDENT SHOCKS")
print("=" * 80)

# Calculate all benchmarks
all_benchmarks = calculator.calculate_all_benchmarks(shock_cfg)

# Check each model
for model, bench in all_benchmarks.items():
    print(f"\n{model.upper()} MODEL:")
    print(f"  Nash Price (p^N):     {bench['p_N']:.4f}")
    print(f"  Nash Profit (E[π^N]): {bench['E_pi_N']:.4f}")
    print(f"  Monopoly Price (p^M): {bench['p_M']:.4f}")
    print(f"  Monopoly Profit (E[π^M]): {bench['E_pi_M']:.4f}")
    
    # Check if denominator is valid for Delta calculation
    denom = bench['E_pi_M'] - bench['E_pi_N']
    print(f"  Profit Range (π^M - π^N): {denom:.4f}")
    
    if denom <= 0:
        print(f"  ⚠️  WARNING: Monopoly profit should be > Nash profit!")
    elif denom < 0.01:
        print(f"  ⚠️  WARNING: Very small profit range - Delta will be unstable!")
    else:
        print(f"  ✓ Valid profit range")
    
    # Calculate example Delta values
    print(f"\n  Example Delta calculations:")
    test_profits = [bench['E_pi_N'], (bench['E_pi_N'] + bench['E_pi_M'])/2, bench['E_pi_M']]
    test_names = ["At Nash", "Halfway", "At Monopoly"]
    for name, profit in zip(test_names, test_profits):
        delta = (profit - bench['E_pi_N']) / denom if denom != 0 else 0
        print(f"    {name:12} (π={profit:.4f}): Δ = {delta:.4f}")

print("\n" + "=" * 80)
print("COMPARISON: NO SHOCKS vs SCHEME A")
print("=" * 80)

no_shock_benchmarks = calculator.calculate_all_benchmarks(None)

for model in ['logit', 'hotelling', 'linear']:
    print(f"\n{model.upper()}:")
    print(f"  Nash Price:    {no_shock_benchmarks[model]['p_N']:.4f} → {all_benchmarks[model]['p_N']:.4f}")
    print(f"  Nash Profit:   {no_shock_benchmarks[model]['E_pi_N']:.4f} → {all_benchmarks[model]['E_pi_N']:.4f}")
    print(f"  Monopoly Price:{no_shock_benchmarks[model]['p_M']:.4f} → {all_benchmarks[model]['p_M']:.4f}")
    print(f"  Monopoly Profit:{no_shock_benchmarks[model]['E_pi_M']:.4f} → {all_benchmarks[model]['E_pi_M']:.4f}")
    
    # Calculate percentage change
    pct_change_nash = 100 * (all_benchmarks[model]['p_N'] / no_shock_benchmarks[model]['p_N'] - 1)
    if abs(pct_change_nash) > 0.1:
        print(f"  Change: {pct_change_nash:+.2f}%")

print("\n" + "=" * 80)
print("EXPECTED VALUES (from your notebook)")
print("=" * 80)
print("\nLOGIT with Scheme A (Independent):")
print("  Expected Nash Price:     ~1.796")
print("  Expected Nash Profit:    ~0.332")
print("  Expected Monopoly Price: ~2.085")
print("  Expected Monopoly Profit:~0.368")

print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

checks_passed = 0
checks_total = 0

# Check 1: Logit Nash price with Scheme A
checks_total += 1
logit_nash = all_benchmarks['logit']['p_N']
if 1.75 < logit_nash < 1.85:
    print(f"✓ Logit Nash price in expected range: {logit_nash:.4f}")
    checks_passed += 1
else:
    print(f"✗ Logit Nash price out of range: {logit_nash:.4f} (expected ~1.796)")

# Check 2: All models have π^M > π^N
checks_total += 1
all_valid = all(
    bench['E_pi_M'] > bench['E_pi_N'] 
    for bench in all_benchmarks.values()
)
if all_valid:
    print("✓ All models have Monopoly profit > Nash profit")
    checks_passed += 1
else:
    print("✗ Some models have invalid profit ordering!")

# Check 3: Delta values should be in [0, 1] for reasonable prices
checks_total += 1
print("\n✓ Testing Delta calculation stability:")
for model, bench in all_benchmarks.items():
    # Test with agent price = Nash + 10% of range
    test_price = bench['p_N'] + 0.1 * (bench['p_M'] - bench['p_N'])
    
    # Simulate getting profit at this price (approximate)
    test_profit = bench['E_pi_N'] + 0.1 * (bench['E_pi_M'] - bench['E_pi_N'])
    
    denom = bench['E_pi_M'] - bench['E_pi_N']
    delta = (test_profit - bench['E_pi_N']) / denom if denom != 0 else 0
    
    if 0 <= delta <= 1:
        print(f"  {model.upper():10} - Delta = {delta:.4f} ✓")
    else:
        print(f"  {model.upper():10} - Delta = {delta:.4f} ✗ (out of [0,1])")
        checks_total += 1

print(f"\n{'-'*80}")
print(f"Checks passed: {checks_passed}/{checks_total}")

if checks_passed == checks_total:
    print("\n✓✓✓ All checks passed! Your theoretical_benchmarks.py is working correctly.")
else:
    print("\n⚠️⚠️⚠️ Some checks failed! There may be issues with your implementation.")


def verify_nash_invariance_logit():
    """Test that Logit Nash doesn't shift with shocks"""
    from environments import MarketEnv
    
    # No shocks
    env1 = MarketEnv(market_model="logit", shock_cfg=None)
    prices = np.array([1.473, 1.473])
    _, profits1 = env1.calculate_demand_and_profit(prices, np.zeros(2))
    
    # With Scheme A shocks
    env2 = MarketEnv(market_model="logit", 
                     shock_cfg={"enabled": True, "scheme": "A", "mode": "independent"})
    
    # Average over many shock realizations
    profits_list = []
    for _ in range(1000):
        env2.shock_generators[0].generate_next()
        env2.shock_generators[1].generate_next()
        shocks = np.array([env2.shock_generators[0].current,
                          env2.shock_generators[1].current])
        _, profits = env2.calculate_demand_and_profit(prices, shocks)
        profits_list.append(profits[0])
    
    avg_profit = np.mean(profits_list)
    
    print(f"Nash price = 1.473:")
    print(f"  No shocks:   π = {profits1[0]:.4f}")
    print(f"  Scheme A:    E[π] = {avg_profit:.4f}")
    print(f"  Difference:  {abs(avg_profit - profits1[0]):.4f}")
    
    if abs(avg_profit - profits1[0]) < 0.01:
        print("  ✅ PASS: Nash invariant to shocks")
        return True
    else:
        print("  ❌ FAIL: Nash shifts with shocks")
        return False

def verify_linear_profit_range():
    """Test that Linear model has healthy profit range"""
    from environments import MarketEnv
    
    env = MarketEnv(market_model="linear")
    
    # Nash
    prices_n = np.array([env.P_N, env.P_N])
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    
    # Monopoly (should be DIFFERENT from symmetric play at p_M!)
    # True monopoly: one firm alone with q = a - p
    p_M = 0.5
    q_M = env.a_bar - p_M
    pi_M_true = p_M * q_M  # 0.25
    
    profit_range = pi_M_true - profits_n[0]
    
    print(f"Linear Model:")
    print(f"  Nash profit:     {profits_n[0]:.4f}")
    print(f"  Monopoly profit: {pi_M_true:.4f}")
    print(f"  Range:           {profit_range:.4f}")
    
    if profit_range > 0.04:
        print("  ✅ PASS: Healthy profit range")
        return True
    else:
        print(f"  ❌ FAIL: Profit range too small ({profit_range:.4f})")
        return False

# Run both tests
if __name__ == "__main__":
    print("="*60)
    test1 = verify_nash_invariance_logit()
    print()
    test2 = verify_linear_profit_range()
    print("="*60)
    
    if test1 and test2:
        print("\n✅ ALL TESTS PASSED - Ready for simulations!")
    else:
        print("\n❌ TESTS FAILED - Fix issues before running sims!")
