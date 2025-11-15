"""
q_vs_ppo.py
"""
import numpy as np
import pandas as pd

from environments import MarketEnvContinuous
from agents import QLearningAgent, PPOAgent

SEED = 99

def run_simulation(model, seed):
    np.random.seed(seed)
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    q_agent = QLearningAgent(env.N, agent_id=0)
    ppo_agent = PPOAgent(agent_id=1, price_min=0.0, price_max=2.0, seed=seed)
    state = env.reset()
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]
    state_ppo = (current_prices[1], current_prices[0])  # own, comp for agent1
    
    profits_history = []
    prices_history = []
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        ppo_price, log_prob, value = ppo_agent.select_action(state_ppo, explore=True)
        
        actions = [q_action, ppo_price]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        ppo_agent.store_transition(state_ppo, ppo_price, rewards[1], log_prob, value, done)
        
        state = next_state
        current_prices = info['prices']
        state_ppo = (current_prices[1], current_prices[0])
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_ppo = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_ppo = np.mean(last_profits[:, 1])
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ppo = (avg_profit_ppo - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_ppo, delta_q, delta_ppo

models = ['logit', 'hotelling', 'linear']
num_runs = 5
results = {}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices_q = []
    avg_prices_ppo = []
    deltas_q = []
    deltas_ppo = []
    for run in range(num_runs):
        seed = SEED + run
        apq, app, dq, dp = run_simulation(model, seed)
        avg_prices_q.append(apq)
        avg_prices_ppo.append(app)
        deltas_q.append(dq)
        deltas_ppo.append(dp)
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price PPO': np.mean(avg_prices_ppo),
        'Delta Q': np.mean(deltas_q),
        'Delta PPO': np.mean(deltas_ppo)
    }

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'PPO Avg. Prices': [round(results[m]['Avg Price PPO'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],  # Repeated for symmetry
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
    'PPO Extra-profits Δ': [round(results[m]['Delta PPO'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_ppo.csv")
print(df)
