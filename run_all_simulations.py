"""
Simple Sequential Runner - Runs each simulation one after another
This avoids subprocess issues by running in the same Python process
"""

import time
from datetime import datetime

print("=" * 80)
print("SIMPLE SEQUENTIAL SIMULATION RUNNER -- Shock Enabled")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

start_time = time.time()
completed = 0
failed = 0

# Q vs Q Simulations
print("\n" + "=" * 80)
print("PART 1/3: Q-LEARNING vs Q-LEARNING")
print("=" * 80)

try:
    print("\n[1/9] Running Q vs Q - Scheme A...")
    import q_vs_q_schemeA
    q_vs_q_schemeA.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[2/9] Running Q vs Q - Scheme B...")
    import q_vs_q_schemeB
    q_vs_q_schemeB.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[3/9] Running Q vs Q - Scheme C...")
    import q_vs_q_schemeC
    q_vs_q_schemeC.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

# Q vs PSO Simulations
print("\n" + "=" * 80)
print("PART 2/3: Q-LEARNING vs PSO")
print("=" * 80)

try:
    print("\n[4/9] Running Q vs PSO - Scheme A...")
    import q_vs_pso_schemeA
    q_vs_pso_schemeA.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[5/9] Running Q vs PSO - Scheme B...")
    import q_vs_pso_schemeB
    q_vs_pso_schemeB.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[6/9] Running Q vs PSO - Scheme C...")
    import q_vs_pso_schemeC
    q_vs_pso_schemeC.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

# PSO vs PSO Simulations
print("\n" + "=" * 80)
print("PART 3/3: PSO vs PSO")
print("=" * 80)

try:
    print("\n[7/9] Running PSO vs PSO - Scheme A...")
    import pso_vs_pso_schemeA
    pso_vs_pso_schemeA.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[8/9] Running PSO vs PSO - Scheme B...")
    import pso_vs_pso_schemeB
    pso_vs_pso_schemeB.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

try:
    print("\n[9/9] Running PSO vs PSO - Scheme C...")
    import pso_vs_pso_schemeC
    pso_vs_pso_schemeC.main()
    completed += 1
    print("✅ Completed")
except Exception as e:
    print(f"❌ Failed: {e}")
    failed += 1

# Final Summary
elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Total simulations: 9")
print(f"Completed: {completed} ✅")
print(f"Failed: {failed} ❌")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

if completed == 9:
    print("\n🎉 All simulations completed successfully!")
    print("\nResults saved to ./results/")
elif completed > 0:
    print(f"\n⚠️  {completed} simulations completed, {failed} failed")
    print("Check errors above for details")
else:
    print("\n❌ All simulations failed - check your environment setup")
