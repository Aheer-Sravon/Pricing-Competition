import sys
import os
import numpy as np
import pandas as pd

# Import the implementations
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import DQNAgent, DDPGAgent
sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, verbose=True):
    """Run a single simulation of DQN vs DDPG."""
    np.random.seed(seed)
    
    # Initialize environment
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    
    # Initialize agents
    dqn_agent = DQNAgent(
        agent_id=0, 
        state_dim=2,  # MUST be 2 for pricing
        action_dim=env.N,
        seed=seed
    )
    
    ddpg_agent = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed
    )
    
    # Reset environment
    state = env.reset()
    
    # Track metrics
    profits_history = []
    prices_history = []
    
    if verbose:
        print(f"\nRunning {model.upper()} model simulation...")
        print(f"Horizon: {env.horizon} steps")
        print(f"Price grid size: {env.N}")
        print(f"Nash price: {env.P_N:.3f}")
        print(f"Monopoly price: {env.P_M:.3f}")
    
    # Main simulation loop
    for t in range(env.horizon):

        # Both agents see state as prices now
        dqn_action = dqn_agent.select_action(state, explore=True)

        # DDPG outputs continuous price directly
        ddpg_state = state.astype(np.float32)
        ddpg_price, ddpg_norm_action = ddpg_agent.select_action(ddpg_state, explore=True)
        
        # Execute actions
        actions = [dqn_action, ddpg_price]
        next_state, rewards, done, info = env.step(actions)
        
        # DQN update
        dqn_agent.remember(state, dqn_action, rewards[0], next_state, done)
        dqn_agent.replay()
        
        # DDPG update
        next_ddpg_state = next_state.astype(np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_norm_action, rewards[1], next_ddpg_state, done)
        ddpg_agent.replay()
        
        # Update exploration rates
        if t % 100 == 0:
            dqn_agent.update_epsilon()
            ddpg_agent.update_epsilon()
        
        # Store history
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        
        # Progress update
        if verbose and (t + 1) % 1000 == 0:
            recent_prices = np.array(prices_history[-100:])
            avg_p_dqn = np.mean(recent_prices[:, 0])
            avg_p_ddpg = np.mean(recent_prices[:, 1])
            print(f"Step {t+1}: DQN price ~{avg_p_dqn:.3f}, DDPG price ~{avg_p_ddpg:.3f}")
    
    # Compute averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_dqn = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_dqn = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Theoretical benchmarks
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]  # Symmetric
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    # Delta: normalized extra profits
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # RPDI: Relative Price Deviation Index
    rpdi_dqn = (avg_price_dqn - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    if verbose:
        print("\nSimulation complete!")
        print(f"DQN: Avg Price {avg_price_dqn:.3f}, Delta {delta_dqn:.3f}, RPDI {rpdi_dqn:.3f}")
        print(f"DDPG: Avg Price {avg_price_ddpg:.3f}, Delta {delta_ddpg:.3f}, RPDI {rpdi_ddpg:.3f}")
    
    return avg_price_dqn, avg_price_ddpg, delta_dqn, delta_ddpg, rpdi_dqn, rpdi_ddpg

models = ['logit', 'hotelling', 'linear']
num_runs = 1
results = {}

for model in models:
    print(f"\n{'='*80}")
    print(f"Model: {model.upper()}")
    print(f"{'='*80}")
    
    avg_prices_dqn = []
    avg_prices_ddpg = []
    deltas_dqn = []
    deltas_ddpg = []
    rpdis_dqn = []
    rpdis_ddpg = []
    
    for run in range(num_runs):
        seed = SEED + run
        ap_dqn, ap_ddpg, d_dqn, d_ddpg, r_dqn, r_ddpg = run_simulation(model, seed, verbose=False)
        avg_prices_dqn.append(ap_dqn)
        avg_prices_ddpg.append(ap_ddpg)
        deltas_dqn.append(d_dqn)
        deltas_ddpg.append(d_ddpg)
        rpdis_dqn.append(r_dqn)
        rpdis_ddpg.append(r_ddpg)
        
        # Log individual run
        print(f"\nRun {run + 1}:")
        print(f"  DQN  -> Delta: {d_dqn:.4f}, RPDI: {r_dqn:.4f}")
        print(f"  DDPG -> Delta: {d_ddpg:.4f}, RPDI: {r_ddpg:.4f}")
    
    results[model] = {
        'Avg Price DQN': np.mean(avg_prices_dqn),
        'Std Price DQN': np.std(avg_prices_dqn),
        'Avg Price DDPG': np.mean(avg_prices_ddpg),
        'Std Price DDPG': np.std(avg_prices_ddpg),
        'Theo Price': MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED).P_N,
        'Delta DQN': np.mean(deltas_dqn),
        'Std Delta DQN': np.std(deltas_dqn),
        'Delta DDPG': np.mean(deltas_ddpg),
        'Std Delta DDPG': np.std(deltas_ddpg),
        'RPDI DQN': np.mean(rpdis_dqn),
        'Std RPDI DQN': np.std(rpdis_dqn),
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
    'DQN Avg Price': [],
    'DQN Delta': [],
    'DQN RPDI': [],
    'DDPG Avg Price': [],
    'DDPG Delta': [],
    'DDPG RPDI': [],
    'Nash Price': []
}

for model in models:
    r = results[model]
    summary_data['Model'].append(model.upper())
    summary_data['DQN Avg Price'].append(f"{r['Avg Price DQN']:.3f}")
    summary_data['DQN Delta'].append(f"{r['Delta DQN']:.3f}")
    summary_data['DQN RPDI'].append(f"{r['RPDI DQN']:.3f}")
    summary_data['DDPG Avg Price'].append(f"{r['Avg Price DDPG']:.3f}")
    summary_data['DDPG Delta'].append(f"{r['Delta DDPG']:.3f}")
    summary_data['DDPG RPDI'].append(f"{r['RPDI DDPG']:.3f}")
    summary_data['Nash Price'].append(f"{r['Theo Price']:.3f}")

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save detailed results
detailed_data = {
    'Model': [m.upper() for m in models],
    'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 3) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 3) for m in models],
    'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 3) for m in models],
    'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 3) for m in models],
    'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 3) for m in models],
    'DQN RPDI': [round(results[m]['RPDI DQN'], 3) for m in models],
    'DDPG RPDI': [round(results[m]['RPDI DDPG'], 3) for m in models]
}

df_detailed = pd.DataFrame(detailed_data)
df_detailed.to_csv("results/dqn_vs_ddpg_2.csv", index=False)
print("\nResults saved to dqn_vs_ddpg_2.csv")

# Print overall averages
print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])
avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])

print("DQN:")
print(f"  Average Delta (Δ): {avg_delta_dqn:.4f}")
print(f"  Average RPDI:      {avg_rpdi_dqn:.4f}")
print("\nDDPG:")
print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}\n")
