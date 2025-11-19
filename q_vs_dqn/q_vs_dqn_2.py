"""
q_vs_dqn.py
"""


import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, DQNAgent

sys.path.pop(0)


SEED = 99

def run_simulation(model, seed):
    np.random.seed(seed)
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    dqn_agent = DQNAgent(agent_id=1, state_dim=2, action_dim=env.N, loss_type='huber', use_double=True, seed=seed)
    
    state = env.reset()
    # CORRECT STATE REPRESENTATION - DQN sees (own_price_idx, competitor_price_idx)
    state_dqn = (state[1], state[0])  # For agent1: own index is state[1], competitor index is state[0]
    
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        dqn_action = dqn_agent.select_action(state_dqn, explore=True)
        
        actions = [q_action, dqn_action]  # agent0 uses q_action, agent1 uses dqn_action
        next_state, rewards, done, info = env.step(actions)
        
        # Q-learning update (agent0)
        q_agent.update(state, q_action, rewards[0], next_state)
        
        # DQN update (agent1) - CORRECT state representation
        next_state_dqn = (next_state[1], next_state[0])  # For agent1: own index is next_state[1], competitor is next_state[0]
        dqn_agent.remember(state_dqn, dqn_action, rewards[1], next_state_dqn, done)
        dqn_agent.replay()
        dqn_agent.update_epsilon()
        
        state = next_state
        state_dqn = next_state_dqn
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Calculate Delta metrics CORRECTLY
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_dqn = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_dqn = np.mean(last_profits[:, 1])
    
    # Calculate industry profits for Delta
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n_industry = np.sum(profits_n)  # Total industry profit at Nash
    
    prices_m = np.array([env.P_M] * 2)  
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m_industry = np.sum(profits_m)  # Total industry profit at Monopoly
    
    # Calculate actual industry profits in simulation
    actual_industry_profit = avg_profit_q + avg_profit_dqn
    
    # Calculate Delta (profit-based)
    delta_q = (avg_profit_q - profits_n[0]) / (profits_m[0] - profits_n[0]) if (profits_m[0] - profits_n[0]) != 0 else 0
    delta_dqn = (avg_profit_dqn - profits_n[1]) / (profits_m[1] - profits_n[1]) if (profits_m[1] - profits_n[1]) != 0 else 0
    
    # Calculate RPDI (pricing-based)
    rpdi_q = (avg_price_q - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi_dqn = (avg_price_dqn - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    return avg_price_q, avg_price_dqn, delta_q, delta_dqn, rpdi_q, rpdi_dqn

models = ['logit', 'hotelling', 'linear']
num_runs = 5
results = {}

# Store individual run results for logging
run_logs = {model: {'delta_q': [], 'delta_dqn': [], 'rpdi_q': [], 'rpdi_dqn': []} for model in models}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices_q = []
    avg_prices_dqn = []
    deltas_q = []
    deltas_dqn = []
    rpdis_q = []
    rpdis_dqn = []
    
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    for run in range(num_runs):
        seed = SEED + run
        apq, apd, dq, dd, rq, rd = run_simulation(model, seed)
        avg_prices_q.append(apq)
        avg_prices_dqn.append(apd)
        deltas_q.append(dq)
        deltas_dqn.append(dd)
        rpdis_q.append(rq)
        rpdis_dqn.append(rd)
        
        # Log individual run results
        print(f"\nRun {run + 1}:")
        print(f"  Q-Learning  -> Delta: {dq:.4f}, RPDI: {rq:.4f}")
        print(f"  DQN         -> Delta: {dd:.4f}, RPDI: {rd:.4f}")
        
        # Store for later access
        run_logs[model]['delta_q'].append(dq)
        run_logs[model]['delta_dqn'].append(dd)
        run_logs[model]['rpdi_q'].append(rq)
        run_logs[model]['rpdi_dqn'].append(rd)
    
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price DQN': np.mean(avg_prices_dqn),
        'Delta Q': np.mean(deltas_q),
        'Delta DQN': np.mean(deltas_dqn),
        'RPDI Q': np.mean(rpdis_q),
        'RPDI DQN': np.mean(rpdis_dqn)
    }

print(f"\n{'='*60}")
print("SUMMARY TABLE")
print(f"{'='*60}\n")

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
    'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],  # Repeated for symmetry
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
    'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 2) for m in models],
    'Q RPDI': [round(results[m]['RPDI Q'], 2) for m in models],
    'DQN RPDI': [round(results[m]['RPDI DQN'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_dqn_2.csv", index=False)
print(df)

# Calculate and print overall averages across all models
print(f"\n{'='*60}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*60}\n")

avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
avg_delta_dqn = np.mean([results[m]['Delta DQN'] for m in models])
avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
avg_rpdi_dqn = np.mean([results[m]['RPDI DQN'] for m in models])

print(f"Q-Learning:")
print(f"  Average Delta (Δ):  {avg_delta_q:.4f}")
print(f"  Average RPDI:       {avg_rpdi_q:.4f}")
print(f"\nDQN:")
print(f"  Average Delta (Δ):  {avg_delta_dqn:.4f}")
print(f"  Average RPDI:       {avg_rpdi_dqn:.4f}")

print(f"\n{'='*60}\n")
