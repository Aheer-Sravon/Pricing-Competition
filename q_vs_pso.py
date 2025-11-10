import numpy as np
import pandas as pd

from agents import PSOAgent, QLearningAgent
from environments import MarketEnvContinuous

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
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_pso = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_pso = np.mean(last_profits[:, 1])
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_pso, delta_q, delta_pso

models = ['logit', 'hotelling', 'linear']
num_runs = 10  # Paper likely uses multiple runs; adjust for computation time
results = {}

for model in models:
    # Get theo price outside the loop since it's constant per model
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices_q = []
    avg_prices_pso = []
    deltas_q = []
    deltas_pso = []
    for run in range(num_runs):
        seed = SEED + run  # Different seeds for each run
        apq, appso, dq, dpso = run_simulation(model, seed)
        avg_prices_q.append(apq)
        avg_prices_pso.append(appso)
        deltas_q.append(dq)
        deltas_pso.append(dpso)
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price PSO': np.mean(avg_prices_pso),
        'Delta Q': np.mean(deltas_q),
        'Delta PSO': np.mean(deltas_pso)
    }

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],  # Repeated for symmetry
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
    'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_pso.csv")
print(df)
