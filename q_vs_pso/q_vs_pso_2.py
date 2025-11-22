import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import PSOAgent, QLearningAgent

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed):
    np.random.seed(seed)
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    pso_agent = PSOAgent(env, agent_id=1)
    state = env.reset()
    profits_history = []
    prices_history = []
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]  # Initial from state
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)  # int index
        pso_agent.update(current_prices[0])  # Update PSO with Q's last price
        pso_price = pso_agent.choose_price()  # float
        actions = [q_action, pso_price]
        next_state, rewards, done, info = env.step(actions)
        q_agent.update(state, q_action, rewards[0], next_state)
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        current_prices = info['prices']
    
    last_prices = np.array(prices_history[-1000:])
    avg_price1 = np.mean(last_prices[:, 0])
    avg_price2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit1 = np.mean(last_profits[:, 0])
    avg_profit2 = np.mean(last_profits[:, 1])
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n1 = profits_n[0]
    pi_n2 = profits_n[1]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m1 = profits_m[0]
    pi_m2 = profits_m[1]
    
    # Calculate Delta (profit-based)
    delta1 = (avg_profit1 - pi_n1) / (pi_m1 - pi_n1) if (pi_m1 - pi_n1) != 0 else 0
    delta2 = (avg_profit2 - pi_n2) / (pi_m2 - pi_n2) if (pi_m2 - pi_n2) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi1 = (avg_price1 - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi2 = (avg_price2 - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    return avg_price1, avg_price2, delta1, delta2, rpdi1, rpdi2

models = ['logit', 'hotelling', 'linear']
num_runs = 50
results = {}

# Store individual run results for logging
run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices1 = []
    avg_prices2 = []
    deltas1 = []
    deltas2 = []
    rpdis1 = []
    rpdis2 = []
    
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    for run in range(num_runs):
        seed = SEED + run
        ap1, ap2, d1, d2, r1, r2 = run_simulation(model, seed)
        avg_prices1.append(ap1)
        avg_prices2.append(ap2)
        deltas1.append(d1)
        deltas2.append(d2)
        rpdis1.append(r1)
        rpdis2.append(r2)
        
        # Log individual run results
        print(f"\nRun {run + 1}:")
        print(f"  Firm 1  -> Delta: {d1:.4f}, RPDI: {r1:.4f}")
        print(f"  Firm 2  -> Delta: {d2:.4f}, RPDI: {r2:.4f}")
        
        # Store for later access
        run_logs[model]['delta1'].append(d1)
        run_logs[model]['delta2'].append(d2)
        run_logs[model]['rpdi1'].append(r1)
        run_logs[model]['rpdi2'].append(r2)
    
    results[model] = {
        'Avg Price Firm 1': np.mean(avg_prices1),
        'Theo Price': p_n,
        'Avg Price Firm 2': np.mean(avg_prices2),
        'Delta 1': np.mean(deltas1),
        'Delta 2': np.mean(deltas2),
        'RPDI 1': np.mean(rpdis1),
        'RPDI 2': np.mean(rpdis2)
    }

print(f"\n{'='*60}")
print("SUMMARY TABLE")
print(f"{'='*60}\n")

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Firm 1 Avg. Prices': [round(results[m]['Avg Price Firm 1'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'Firm 2 Avg. Prices': [round(results[m]['Avg Price Firm 2'], 2) for m in models],
    'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
    'Firm 1 Extra-profits Δ': [round(results[m]['Delta 1'], 2) for m in models],
    'Firm 2 Extra-profits Δ': [round(results[m]['Delta 2'], 2) for m in models],
    'Firm 1 RPDI': [round(results[m]['RPDI 1'], 2) for m in models],
    'Firm 2 RPDI': [round(results[m]['RPDI 2'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_pso.csv", index=False)
print(df)

# Calculate and print overall averages across all models
print(f"\n{'='*60}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*60}\n")

avg_delta1 = np.mean([results[m]['Delta 1'] for m in models])
avg_delta2 = np.mean([results[m]['Delta 2'] for m in models])
avg_rpdi1 = np.mean([results[m]['RPDI 1'] for m in models])
avg_rpdi2 = np.mean([results[m]['RPDI 2'] for m in models])

print(f"Firm 1 (Q-Learning):")
print(f"  Average Delta (Δ):  {avg_delta1:.4f}")
print(f"  Average RPDI:       {avg_rpdi1:.4f}")
print(f"\nFirm 2 (PSO):")
print(f"  Average Delta (Δ):  {avg_delta2:.4f}")
print(f"  Average RPDI:       {avg_rpdi2:.4f}")

print(f"\n{'='*60}\n")
