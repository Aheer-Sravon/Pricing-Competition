import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import DQNAgent, PSOAgent

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg):
    """Run PSO vs DQN simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    # Compute theoretical benchmarks directly
    p_n = env.P_N
    p_m = env.P_M
    prices_n = np.array([p_n, p_n])
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    prices_m = np.array([p_m, p_m])
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    # Initialize PSO agent (agent 0)
    pso_agent = PSOAgent(
        env, 
        agent_id=0, 
        price_min=env.price_grid.min(), 
        price_max=env.price_grid.max()
    )
    
    # Initialize DQN agent (agent 1)
    dqn_agent = DQNAgent(
        agent_id=1, 
        state_dim=2, 
        action_dim=env.N, 
        seed=seed
    )
    
    # Reset environment - state is now actual prices
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        # PSO updates with DQN's last price (state[1])
        pso_agent.update(state[1])
        pso_price = pso_agent.choose_price()  # Continuous price
        
        # DQN selects action based on state (prices)
        dqn_action = dqn_agent.select_action(state, explore=True)  # Discrete index
        
        # Execute actions (continuous price + discrete index)
        actions = [pso_price, dqn_action]
        next_state, rewards, done, info = env.step(actions)
        
        # Update DQN (state is already prices, no conversion needed)
        dqn_agent.remember(state, dqn_action, rewards[1], next_state, done)
        dqn_agent.replay()
        
        # Decay exploration
        if t % 100 == 0:
            dqn_agent.update_epsilon()
        
        # Update state
        state = next_state
        
        # Record history
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_pso = np.mean(last_prices[:, 0])
    avg_price_dqn = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_pso = np.mean(last_profits[:, 0])
    avg_profit_dqn = np.mean(last_profits[:, 1])
    
    # Calculate Delta (profit-based)
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_pso = (avg_price_pso - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_dqn = (avg_price_dqn - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_pso, avg_price_dqn, delta_pso, delta_dqn, rpdi_pso, rpdi_dqn, p_n

shock_cfg = {
    'enabled': False
}

print("=" * 80)
print("PSO vs DQN - NO SHOCKS")
print("=" * 80)

models = ['logit', 'hotelling', 'linear']
num_runs = 5
results = {}

# Store individual run results for logging
run_logs = {model: {'delta_pso': [], 'delta_dqn': [], 'rpdi_pso': [], 'rpdi_dqn': []} for model in models}

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    avg_prices_pso = []
    avg_prices_dqn = []
    deltas_pso = []
    deltas_dqn = []
    rpdis_pso = []
    rpdis_dqn = []
    theo_prices = []
    
    for run in range(num_runs):
        seed = SEED + run
        ap_pso, ap_dqn, d_pso, d_dqn, r_pso, r_dqn, p_n = run_simulation(model, seed, shock_cfg)
        avg_prices_pso.append(ap_pso)
        avg_prices_dqn.append(ap_dqn)
        deltas_pso.append(d_pso)
        deltas_dqn.append(d_dqn)
        rpdis_pso.append(r_pso)
        rpdis_dqn.append(r_dqn)
        theo_prices.append(p_n)
        
        # Log individual run results
        print(f"\nRun {run + 1}:")
        print(f"  PSO -> Delta: {d_pso:.4f}, RPDI: {r_pso:.4f}")
        print(f"  DQN -> Delta: {d_dqn:.4f}, RPDI: {r_dqn:.4f}")
        
        # Store for later access
        run_logs[model]['delta_pso'].append(d_pso)
        run_logs[model]['delta_dqn'].append(d_dqn)
        run_logs[model]['rpdi_pso'].append(r_pso)
        run_logs[model]['rpdi_dqn'].append(r_dqn)
    
    results[model] = {
        'Avg Price PSO': np.mean(avg_prices_pso),
        'Theo Price': np.mean(theo_prices),
        'Avg Price DQN': np.mean(avg_prices_dqn),
        'Delta PSO': np.mean(deltas_pso),
        'Delta DQN': np.mean(deltas_dqn),
        'RPDI PSO': np.mean(rpdis_pso),
        'RPDI DQN': np.mean(rpdis_dqn)
    }
    
    print(f"\n  Model Average: PSO Δ = {results[model]['Delta PSO']:.3f}, DQN Δ = {results[model]['Delta DQN']:.3f}")

print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}\n")

data = {
    'Model': [m.upper() for m in models],
    'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
    'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
    'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models],
    'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 2) for m in models],
    'PSO RPDI': [round(results[m]['RPDI PSO'], 2) for m in models],
    'DQN RPDI': [round(results[m]['RPDI DQN'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/pso_vs_dqn_2.csv", index=False)
print(df.to_string(index=False))

# Calculate and print overall averages across all models
print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta_pso = np.mean([results[m]['Delta PSO'] for m in models])
avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
avg_rpdi_pso = np.mean([results[m]['RPDI PSO'] for m in models])
avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])

print("PSO Agent:")
print(f"  Average Delta (Δ):  {avg_delta_pso:.4f}")
print(f"  Average RPDI:       {avg_rpdi_pso:.4f}")
print("\nDQN Agent:")
print(f"  Average Delta (Δ):  {avg_delta_dqn:.4f}")
print(f"  Average RPDI:       {avg_rpdi_dqn:.4f}")

print(f"\n{'='*80}")
print("[Results saved to ./results/pso_vs_dqn_2.csv]")
print(f"{'='*80}\n")
