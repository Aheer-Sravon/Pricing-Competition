import numpy as np
import pandas as pd

from environments import MarketEnvContinuous
from agents import QLearningAgent

SEED = 99

def run_simulation(model, seed):
    np.random.seed(seed)
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    agents = [QLearningAgent(env.N, agent_id=0), QLearningAgent(env.N, agent_id=1)]
    state = env.reset()
    profits_history = []
    prices_history = []
    for t in range(env.horizon):
        actions = [agents[0].choose_action(state), agents[1].choose_action(state)]
        next_state, rewards, done, info = env.step(actions)
        agents[0].update(state, actions[0], rewards[0], next_state)
        agents[1].update(state, actions[1], rewards[1], next_state)
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price1 = np.mean(last_prices[:, 0])
    avg_price2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit = np.mean(last_profits[:, 0])  # Symmetric, use firm 1
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    delta = (avg_profit - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price1, avg_price2, delta

models = ['logit', 'hotelling', 'linear']
num_runs = 10  # Paper likely uses multiple runs; adjust for computation time
results = {}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices1 = []
    avg_prices2 = []
    deltas = []
    for run in range(num_runs):
        seed = SEED + run  # Different seeds for each run
        ap1, ap2, d = run_simulation(model, seed)
        avg_prices1.append(ap1)
        avg_prices2.append(ap2)
        deltas.append(d)
    results[model] = {
        'Avg Price Firm 1': np.mean(avg_prices1),
        'Theo Price': p_n,
        'Avg Price Firm 2': np.mean(avg_prices2),
        'Delta': np.mean(deltas)
    }

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Firm 1 Avg. Prices': [round(results[m]['Avg Price Firm 1'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'Firm 2 Avg. Prices': [round(results[m]['Avg Price Firm 2'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],  # Repeated for symmetry
    'Extra-profits Δ': [round(results[m]['Delta'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_q.csv")
print(df)
