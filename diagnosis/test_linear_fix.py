"""
Test if theoretical_benchmarks.py was properly fixed
Run this in your project directory where theoretical_benchmarks.py exists
"""

from theoretical_benchmarks import TheoreticalBenchmarks

print("=" * 70)
print("TESTING: theoretical_benchmarks.py Linear Model Fix")
print("=" * 70)

calc = TheoreticalBenchmarks()
linear = calc.calculate_linear_benchmarks()

print(f"\nLinear Model Benchmarks:")
print(f"  Nash price:      p^N = {linear['p_N']:.4f}")
print(f"  Nash profit:     π^N = {linear['E_pi_N']:.4f}")
print(f"  Monopoly price:  p^M = {linear['p_M']:.4f}")
print(f"  Monopoly profit: π^M = {linear['E_pi_M']:.4f}")

profit_range = linear['E_pi_M'] - linear['E_pi_N']
print(f"  Profit range:    π^M - π^N = {profit_range:.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

if abs(linear['E_pi_M'] - 0.25) < 0.001:
    print("✅ FIXED! Monopoly profit = 0.25")
    print("   Profit range = {:.4f} (healthy)".format(profit_range))
    print("\n   Your theoretical_benchmarks.py is CORRECT!")
    print("   You can now re-run your simulations.")
    
elif abs(linear['E_pi_M'] - 0.20) < 0.001:
    print("❌ NOT FIXED! Monopoly profit = 0.20 (should be 0.25)")
    print("   Profit range = {:.4f} (too small)".format(profit_range))
    print("\n   The fix was NOT applied correctly.")
    print("\n   In theoretical_benchmarks.py, find this line:")
    print("   q_M = (a_bar - p_M - d * (a_bar - p_M)) / denominator")
    print("\n   Replace with:")
    print("   q_M = a_bar - p_M")
    print("\n   Make sure you:")
    print("   1. Saved the file after editing")
    print("   2. Don't have multiple copies of theoretical_benchmarks.py")
    print("   3. Edited the correct file in your project directory")
    
else:
    print(f"❓ UNEXPECTED: Monopoly profit = {linear['E_pi_M']:.4f}")
    print("   Neither 0.20 nor 0.25 - please review the code")

print("=" * 70)
