import sys
import os
import numpy as np
import pandas as pd

# Import the new DQN implementation
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, DQNAgent
sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, verbose=True):
    """Run a single simulation of Q-Learning vs DQN."""
    np.random.seed(seed)
    
    # Initialize environment
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    
    # Initialize agents
    q_agent = QLearningAgent(
        env.N,
        agent_id=0,
        price_grid=env.price_grid
    )
    
    # Initialize DQN with proper parameters
    dqn_agent = DQNAgent(
        agent_id=1, 
        state_dim=2,  # MUST be 2 for pricing
        action_dim=env.N,
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
        q_action = q_agent.choose_action(state)
        
        dqn_action = dqn_agent.select_action(state, explore=True)
        
        # Execute actions
        actions = [q_action, dqn_action]
        next_state, rewards, done, info = env.step(actions)
        
        # Both agents use same state representation
        q_agent.update(state, q_action, rewards[0], next_state)
        dqn_agent.remember(state, dqn_action, rewards[1], next_state, done)
        
        # Train DQN
        dqn_agent.replay()
        
        # Update exploration rate
        if t % 100 == 0:
            dqn_agent.update_epsilon()
        
        # Store history
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        
        # Progress update
        if verbose and (t + 1) % 1000 == 0:
            recent_prices = np.array(prices_history[-100:])
            avg_p_q = np.mean(recent_prices[:, 0])
            avg_p_dqn = np.mean(recent_prices[:, 1])
            print(f"  Step {t+1}/{env.horizon}: Q price={avg_p_q:.3f}, DQN price={avg_p_dqn:.3f}, ε={dqn_agent.epsilon:.3f}")
    
    # Calculate final metrics (last 1000 steps)
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_dqn = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_dqn = np.mean(last_profits[:, 1])
    
    # Calculate Nash and Monopoly benchmarks
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    
    prices_m = np.array([env.P_M] * 2)  
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    
    # Calculate Delta (profit-based collusion metric)
    delta_q = (avg_profit_q - profits_n[0]) / (profits_m[0] - profits_n[0]) if (profits_m[0] - profits_n[0]) != 0 else 0
    delta_dqn = (avg_profit_dqn - profits_n[1]) / (profits_m[1] - profits_n[1]) if (profits_m[1] - profits_n[1]) != 0 else 0
    
    # Calculate RPDI (price-based collusion metric)
    rpdi_q = (avg_price_q - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi_dqn = (avg_price_dqn - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    return avg_price_q, avg_price_dqn, delta_q, delta_dqn, rpdi_q, rpdi_dqn


models = ['logit', 'hotelling', 'linear']
num_runs = 1
results = {}

# Store individual run results
run_logs = {model: {'delta_q': [], 'delta_dqn': [], 'rpdi_q': [], 'rpdi_dqn': []} for model in models}

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    # Get theoretical Nash price
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    p_m = env_temp.P_M
    
    print(f"Nash equilibrium price: {p_n:.3f}")
    print(f"Monopoly price: {p_m:.3f}")
    
    avg_prices_q = []
    avg_prices_dqn = []
    deltas_q = []
    deltas_dqn = []
    rpdis_q = []
    rpdis_dqn = []
    
    for run in range(num_runs):
        seed = SEED + run
        print(f"\nRun {run + 1}/{num_runs} (seed={seed})")
        
        apq, apd, dq, dd, rq, rd = run_simulation(model, seed, verbose=(run == 0))
        
        avg_prices_q.append(apq)
        avg_prices_dqn.append(apd)
        deltas_q.append(dq)
        deltas_dqn.append(dd)
        rpdis_q.append(rq)
        rpdis_dqn.append(rd)
        
        # Log results
        print(f"\nResults for Run {run + 1}:")
        print(f"  Q-Learning  -> Price: {apq:.3f}, Delta: {dq:.4f}, RPDI: {rq:.4f}")
        print(f"  DQN         -> Price: {apd:.3f}, Delta: {dd:.4f}, RPDI: {rd:.4f}")
        
        # Store for analysis
        run_logs[model]['delta_q'].append(dq)
        run_logs[model]['delta_dqn'].append(dd)
        run_logs[model]['rpdi_q'].append(rq)
        run_logs[model]['rpdi_dqn'].append(rd)
    
    # Store average results
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Std Price Q': np.std(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price DQN': np.mean(avg_prices_dqn),
        'Std Price DQN': np.std(avg_prices_dqn),
        'Delta Q': np.mean(deltas_q),
        'Std Delta Q': np.std(deltas_q),
        'Delta DQN': np.mean(deltas_dqn),
        'Std Delta DQN': np.std(deltas_dqn),
        'RPDI Q': np.mean(rpdis_q),
        'Std RPDI Q': np.std(rpdis_q),
        'RPDI DQN': np.mean(rpdis_dqn),
        'Std RPDI DQN': np.std(rpdis_dqn)
    }

# Print summary
print(f"\n{'='*80}")
print("SUMMARY RESULTS")
print(f"{'='*80}\n")

# Create summary dataframe
summary_data = {
    'Model': [],
    'Q-Learning Avg Price': [],
    'Q-Learning Delta': [],
    'Q-Learning RPDI': [],
    'DQN Avg Price': [],
    'DQN Delta': [],
    'DQN RPDI': [],
    'Nash Price': []
}

for model in models:
    r = results[model]
    summary_data['Model'].append(model.upper())
    summary_data['Q-Learning Avg Price'].append(f"{r['Avg Price Q']:.3f}")
    summary_data['Q-Learning Delta'].append(f"{r['Delta Q']:.3f}")
    summary_data['Q-Learning RPDI'].append(f"{r['RPDI Q']:.3f}")
    summary_data['DQN Avg Price'].append(f"{r['Avg Price DQN']:.3f}")
    summary_data['DQN Delta'].append(f"{r['Delta DQN']:.3f}")
    summary_data['DQN RPDI'].append(f"{r['RPDI DQN']:.3f}")
    summary_data['Nash Price'].append(f"{r['Theo Price']:.3f}")

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save detailed results
detailed_data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 3) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 3) for m in models],
    'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 3) for m in models],
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 3) for m in models],
    'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 3) for m in models],
    'Q RPDI': [round(results[m]['RPDI Q'], 3) for m in models],
    'DQN RPDI': [round(results[m]['RPDI DQN'], 3) for m in models]
}

df_detailed = pd.DataFrame(detailed_data)
df_detailed.to_csv("results/q_vs_dqn_2.csv", index=False)
print(f"\nResults saved to q_vs_dqn_fixed.csv")

# Print overall averages
print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])

print(f"Q-Learning:")
print(f"  Average Delta (Δ): {avg_delta_q:.4f}")
print(f"  Average RPDI:      {avg_rpdi_q:.4f}")
print(f"\nDQN (Fixed):")
print(f"  Average Delta (Δ): {avg_delta_dqn:.4f}")
print(f"  Average RPDI:      {avg_rpdi_dqn:.4f}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}\n")
