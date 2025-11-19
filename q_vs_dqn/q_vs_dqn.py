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
    state_dqn = (state[1], state[0])  # own, comp for agent1
    
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        dqn_action = dqn_agent.select_action(state_dqn, explore=True)
        
        actions = [q_action, dqn_action]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        next_state_dqn = (next_state[1], next_state[0])
        dqn_agent.remember(state_dqn, dqn_action, rewards[1], next_state_dqn, done)
        dqn_agent.replay()
        dqn_agent.update_epsilon()
        
        state = next_state
        state_dqn = next_state_dqn
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_dqn = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_dqn = np.mean(last_profits[:, 1])
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_dqn = (avg_profit_dqn - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_dqn, delta_q, delta_dqn

models = ['logit', 'hotelling', 'linear']
num_runs = 5
results = {}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices_q = []
    avg_prices_dqn = []
    deltas_q = []
    deltas_dqn = []
    for run in range(num_runs):
        seed = SEED + run
        apq, apd, dq, dd = run_simulation(model, seed)
        avg_prices_q.append(apq)
        avg_prices_dqn.append(apd)
        deltas_q.append(dq)
        deltas_dqn.append(dd)
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price DQN': np.mean(avg_prices_dqn),
        'Delta Q': np.mean(deltas_q),
        'Delta DQN': np.mean(deltas_dqn)
    }

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'DQN Avg. Prices': [round(results[m]['Avg Price DQN'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],  # Repeated for symmetry
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
    'DQN Extra-profits Δ': [round(results[m]['Delta DQN'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_dqn.csv")
print(df)
