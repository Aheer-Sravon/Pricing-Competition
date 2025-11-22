import sys
import os
import numpy as np
import pandas as pd

# Import the implementations
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import PSOAgent, DDPGAgent
sys.path.pop(0)

SEED = 99
HORIZON = 10000  # Simulation horizon

def run_simulation(model, seed, verbose=True):
    """Run a single simulation of PSO vs DDPG."""
    np.random.seed(seed)
    
    # Initialize environment
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    
    # Initialize agents
    pso_agent = PSOAgent(env, agent_id=0)
    
    # Initialize DDPG with proper parameters
    ddpg_agent = DDPGAgent(
        agent_id=1, 
        state_dim=2,  # MUST be 2 for pricing
        action_dim=1,
        seed=seed
    )
    
    # Reset environment
    state = env.reset()
    
    # Track metrics
    profits_history = []
    prices_history = []
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]  # Initial from state
    
    if verbose:
        print(f"\nRunning {model.upper()} model simulation...")
        print(f"Horizon: {HORIZON} steps")
        print(f"Price grid size: {env.N}")
        print(f"Nash price: {env.P_N:.3f}")
        print(f"Monopoly price: {env.P_M:.3f}")
    
    # Main simulation loop
    for t in range(HORIZON):
        # PSO updates with DDPG's last price
        pso_agent.update(current_prices[1])
        pso_action = pso_agent.choose_price()  # float
        
        # DDPG sees state as (own_price_idx, competitor_price_idx)
        ddpg_state = np.array([state[1], state[0]], dtype=np.float32)
        ddpg_action_raw = ddpg_agent.select_action(ddpg_state, explore=True)
        
        # Map continuous action [-1, 1] to discrete price index
        ddpg_action = int((ddpg_action_raw[0] + 1) / 2 * (env.N - 1))
        ddpg_action = np.clip(ddpg_action, 0, env.N - 1)
        
        # Execute actions
        actions = [pso_action, ddpg_action]
        next_state, rewards, done, info = env.step(actions)
        
        # DDPG update with correct state representation
        next_ddpg_state = np.array([next_state[1], next_state[0]], dtype=np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_action_raw, rewards[1], next_ddpg_state, done)
        
        # Train DDPG
        ddpg_agent.replay()
        
        # Update exploration rate
        if t % 100 == 0:
            ddpg_agent.update_epsilon()
        
        # Store history
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        current_prices = info['prices']
        
        # Progress update
        if verbose and (t + 1) % 1000 == 0:
            recent_prices = np.array(prices_history[-100:])
            avg_p_pso = np.mean(recent_prices[:, 0])
            avg_p_ddpg = np.mean(recent_prices[:, 1])
            print(f"Step {t+1}: PSO price ~{avg_p_pso:.3f}, DDPG price ~{avg_p_ddpg:.3f}")
    
    # Compute averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_pso = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_pso = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Theoretical benchmarks
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]  # Symmetric
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    # Delta: normalized extra profits
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # RPDI: Relative Price Deviation Index
    rpdi_pso = (avg_price_pso - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    if verbose:
        print(f"\nSimulation complete!")
        print(f"PSO: Avg Price {avg_price_pso:.3f}, Delta {delta_pso:.3f}, RPDI {rpdi_pso:.3f}")
        print(f"DDPG: Avg Price {avg_price_ddpg:.3f}, Delta {delta_ddpg:.3f}, RPDI {rpdi_ddpg:.3f}")
    
    return avg_price_pso, avg_price_ddpg, delta_pso, delta_ddpg, rpdi_pso, rpdi_ddpg

models = ['logit', 'hotelling', 'linear']
num_runs = 50
results = {}

for model in models:
    print(f"\n{'='*80}")
    print(f"Model: {model.upper()}")
    print(f"{'='*80}")
    
    avg_prices_pso = []
    avg_prices_ddpg = []
    deltas_pso = []
    deltas_ddpg = []
    rpdis_pso = []
    rpdis_ddpg = []
    
    for run in range(num_runs):
        seed = SEED + run
        ap_pso, ap_ddpg, d_pso, d_ddpg, r_pso, r_ddpg = run_simulation(model, seed, verbose=False)
        avg_prices_pso.append(ap_pso)
        avg_prices_ddpg.append(ap_ddpg)
        deltas_pso.append(d_pso)
        deltas_ddpg.append(d_ddpg)
        rpdis_pso.append(r_pso)
        rpdis_ddpg.append(r_ddpg)
        
        # Log individual run
        print(f"\nRun {run + 1}:")
        print(f"  PSO  -> Delta: {d_pso:.4f}, RPDI: {r_pso:.4f}")
        print(f"  DDPG  -> Delta: {d_ddpg:.4f}, RPDI: {r_ddpg:.4f}")
    
    results[model] = {
        'Avg Price PSO': np.mean(avg_prices_pso),
        'Std Price PSO': np.std(avg_prices_pso),
        'Avg Price DDPG': np.mean(avg_prices_ddpg),
        'Std Price DDPG': np.std(avg_prices_ddpg),
        'Theo Price': MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED).P_N,
        'Delta PSO': np.mean(deltas_pso),
        'Std Delta PSO': np.std(deltas_pso),
        'Delta DDPG': np.mean(deltas_ddpg),
        'Std Delta DDPG': np.std(deltas_ddpg),
        'RPDI PSO': np.mean(rpdis_pso),
        'Std RPDI PSO': np.std(rpdis_pso),
        'RPDI DDPG': np.mean(rpdis_ddpg),
        'Std RPDI DDPG': np.std(rpdis_ddpg)
    }

# Print summary
print(f"\n{'='*80}")
print("SUMMARY RESULTS")
print(f"{'='*80}\n")

# Create summary dataframe
summary_data = {
    'Model': [],
    'PSO Avg Price': [],
    'PSO Delta': [],
    'PSO RPDI': [],
    'DDPG Avg Price': [],
    'DDPG Delta': [],
    'DDPG RPDI': [],
    'Nash Price': []
}

for model in models:
    r = results[model]
    summary_data['Model'].append(model.upper())
    summary_data['PSO Avg Price'].append(f"{r['Avg Price PSO']:.3f} ± {r['Std Price PSO']:.3f}")
    summary_data['PSO Delta'].append(f"{r['Delta PSO']:.3f} ± {r['Std Delta PSO']:.3f}")
    summary_data['PSO RPDI'].append(f"{r['RPDI PSO']:.3f} ± {r['Std RPDI PSO']:.3f}")
    summary_data['DDPG Avg Price'].append(f"{r['Avg Price DDPG']:.3f} ± {r['Std Price DDPG']:.3f}")
    summary_data['DDPG Delta'].append(f"{r['Delta DDPG']:.3f} ± {r['Std Delta DDPG']:.3f}")
    summary_data['DDPG RPDI'].append(f"{r['RPDI DDPG']:.3f} ± {r['Std RPDI DDPG']:.3f}")
    summary_data['Nash Price'].append(f"{r['Theo Price']:.3f}")

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save detailed results
detailed_data = {
    'Model': [m.upper() for m in models],
    'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 3) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 3) for m in models],
    'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 3) for m in models],
    'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 3) for m in models],
    'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 3) for m in models],
    'PSO RPDI': [round(results[m]['RPDI PSO'], 3) for m in models],
    'DDPG RPDI': [round(results[m]['RPDI DDPG'], 3) for m in models]
}

df_detailed = pd.DataFrame(detailed_data)
df_detailed.to_csv("results/pso_vs_ddpg_2.csv", index=False)
print(f"\nResults saved to pso_vs_ddpg_2.csv")

# Print overall averages
print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta_pso = np.mean([results[m]['Delta PSO'] for m in models])
avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
avg_rpdi_pso = np.mean([results[m]['RPDI PSO'] for m in models])
avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])

print(f"PSO:")
print(f"  Average Delta (Δ): {avg_delta_pso:.4f}")
print(f"  Average RPDI:      {avg_rpdi_pso:.4f}")
print(f"\nDDPG:")
print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")
