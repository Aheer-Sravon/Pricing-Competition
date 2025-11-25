import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import DDPGAgent

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg):
    """Run DDPG vs DDPG simulation"""
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
    
    # Get price bounds
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize DDPG agents
    ddpg_agent1 = DDPGAgent(
        agent_id=0,
        state_dim=2,
        action_dim=1,
        seed=seed,
        price_min=price_min,
        price_max=price_max
    )
    
    ddpg_agent2 = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed + 1000,  # Different seed for diversity
        price_min=price_min,
        price_max=price_max
    )
    
    # Reset environment - state is actual prices
    state = env.reset()
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        # Both agents select continuous prices
        price1, norm_action1 = ddpg_agent1.select_action(state.astype(np.float32), explore=True)
        price2, norm_action2 = ddpg_agent2.select_action(state.astype(np.float32), explore=True)
        
        # Execute actions (both continuous prices)
        actions = [price1, price2]
        next_state, rewards, done, info = env.step(actions)
        
        # Convert states to float32 for PyTorch
        state_float = state.astype(np.float32)
        next_state_float = next_state.astype(np.float32)
        
        # Update both agents
        ddpg_agent1.remember(state_float, norm_action1, rewards[0], next_state_float, done)
        ddpg_agent1.replay()
        
        ddpg_agent2.remember(state_float, norm_action2, rewards[1], next_state_float, done)
        ddpg_agent2.replay()
        
        # Decay exploration
        if t % 100 == 0:
            ddpg_agent1.update_epsilon()
            ddpg_agent2.update_epsilon()
        
        # Update state
        state = next_state
        
        # Record history
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_ddpg1 = np.mean(last_prices[:, 0])
    avg_price_ddpg2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_ddpg1 = np.mean(last_profits[:, 0])
    avg_profit_ddpg2 = np.mean(last_profits[:, 1])
    
    # Calculate Delta (profit-based)
    delta_ddpg1 = (avg_profit_ddpg1 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg2 = (avg_profit_ddpg2 - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_ddpg1 = (avg_price_ddpg1 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg2 = (avg_price_ddpg2 - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_ddpg1, avg_price_ddpg2, delta_ddpg1, delta_ddpg2, rpdi_ddpg1, rpdi_ddpg2, p_n

shock_cfg = {
    'enabled': False
}

print("=" * 80)
print("DDPG vs DDPG - NO SHOCKS")
print("=" * 80)

models = ['logit', 'hotelling', 'linear']
num_runs = 1
results = {}

# Store individual run results for logging
run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    avg_prices_ddpg1 = []
    avg_prices_ddpg2 = []
    deltas_ddpg1 = []
    deltas_ddpg2 = []
    rpdis_ddpg1 = []
    rpdis_ddpg2 = []
    theo_prices = []
    
    for run in range(num_runs):
        seed = SEED + run
        ap1, ap2, d1, d2, r1, r2, p_n = run_simulation(model, seed, shock_cfg)
        avg_prices_ddpg1.append(ap1)
        avg_prices_ddpg2.append(ap2)
        deltas_ddpg1.append(d1)
        deltas_ddpg2.append(d2)
        rpdis_ddpg1.append(r1)
        rpdis_ddpg2.append(r2)
        theo_prices.append(p_n)
        
        # Log individual run results
        print(f"\nRun {run + 1}:")
        print(f"  DDPG 1 -> Delta: {d1:.4f}, RPDI: {r1:.4f}")
        print(f"  DDPG 2 -> Delta: {d2:.4f}, RPDI: {r2:.4f}")
        
        # Store for later access
        run_logs[model]['delta1'].append(d1)
        run_logs[model]['delta2'].append(d2)
        run_logs[model]['rpdi1'].append(r1)
        run_logs[model]['rpdi2'].append(r2)
    
    results[model] = {
        'Avg Price DDPG1': np.mean(avg_prices_ddpg1),
        'Theo Price': np.mean(theo_prices),
        'Avg Price DDPG2': np.mean(avg_prices_ddpg2),
        'Delta DDPG1': np.mean(deltas_ddpg1),
        'Delta DDPG2': np.mean(deltas_ddpg2),
        'RPDI DDPG1': np.mean(rpdis_ddpg1),
        'RPDI DDPG2': np.mean(rpdis_ddpg2)
    }
    
    print(f"\n  Model Average: DDPG1 Δ = {results[model]['Delta DDPG1']:.3f}, DDPG2 Δ = {results[model]['Delta DDPG2']:.3f}")

print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}\n")

data = {
    'Model': [m.upper() for m in models],
    'DDPG1 Avg. Prices': [round(results[m]['Avg Price DDPG1'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'DDPG2 Avg. Prices': [round(results[m]['Avg Price DDPG2'], 2) for m in models],
    'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
    'DDPG1 Extra-profits Δ': [round(results[m]['Delta DDPG1'], 2) for m in models],
    'DDPG2 Extra-profits Δ': [round(results[m]['Delta DDPG2'], 2) for m in models],
    'DDPG1 RPDI': [round(results[m]['RPDI DDPG1'], 2) for m in models],
    'DDPG2 RPDI': [round(results[m]['RPDI DDPG2'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/ddpg_vs_ddpg_2.csv", index=False)
print(df.to_string(index=False))

# Calculate and print overall averages across all models
print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta1 = np.mean([results[m]['Delta DDPG1'] for m in models])
avg_delta2 = np.mean([results[m]['Delta DDPG2'] for m in models])
avg_rpdi1 = np.mean([results[m]['RPDI DDPG1'] for m in models])
avg_rpdi2 = np.mean([results[m]['RPDI DDPG2'] for m in models])

print("DDPG Agent 1:")
print(f"  Average Delta (Δ):  {avg_delta1:.4f}")
print(f"  Average RPDI:       {avg_rpdi1:.4f}")
print("\nDDPG Agent 2:")
print(f"  Average Delta (Δ):  {avg_delta2:.4f}")
print(f"  Average RPDI:       {avg_rpdi2:.4f}")

print(f"\n{'='*80}")
print("[Results saved to ./results/ddpg_vs_ddpg_2.csv]")
print(f"{'='*80}\n")
