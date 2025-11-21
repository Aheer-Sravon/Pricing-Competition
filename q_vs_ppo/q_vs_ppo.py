import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, PPOAgent
sys.path.pop(0)

SEED = 99

def run_simulation(model, seed):
    np.random.seed(seed)
    
    # INCREASE HORIZON for better PPO learning
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    
    # Use shorter horizon for PPO to update more frequently
    ppo_horizon = 512  # Reduced from 2048 for more frequent updates
    q_agent = QLearningAgent(env.N, agent_id=0)
    ppo_agent = PPOAgent(
        agent_id=1, 
        price_min=0.0, 
        price_max=2.0, 
        horizon=ppo_horizon,  # Custom horizon
        seed=seed
    )
    
    state = env.reset()
    current_prices = [env.price_grid[state[0]], env.price_grid[state[1]]]
    state_ppo = (current_prices[1], current_prices[0])  # own, comp for agent1
    
    profits_history = []
    prices_history = []
    
    update_count = 0
    
    for t in range(env.horizon):
        q_action = q_agent.choose_action(state)
        ppo_price, log_prob, value = ppo_agent.select_action(state_ppo, explore=True)
        
        actions = [q_action, ppo_price]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action, rewards[0], next_state)
        ppo_agent.store_transition(state_ppo, ppo_price, rewards[1], log_prob, value, done)
        
        # MORE FREQUENT UPDATES with shorter horizon
        if len(ppo_agent.states) >= ppo_agent.horizon:
            update_stats = ppo_agent.update(state_ppo)
            update_count += 1
            if update_count % 5 == 0:  # Print every 5th update
                print(f"Step {t}: PPO Update {update_count} (actor_loss: {update_stats['actor_loss']:.4f})")
        
        state = next_state
        current_prices = info['prices']
        state_ppo = (current_prices[1], current_prices[0])
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Final update with any remaining data
    if len(ppo_agent.states) > 0:
        update_stats = ppo_agent.update(state_ppo)
        update_count += 1
        print(f"Final PPO update {update_count} (actor_loss: {update_stats['actor_loss']:.4f})")
    
    print(f"Total PPO updates: {update_count}")
    
    # Use last 20% of data for evaluation (more stable)
    eval_start = int(0.8 * len(prices_history))
    last_prices = np.array(prices_history[eval_start:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_ppo = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[eval_start:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_ppo = np.mean(last_profits[:, 1])
    
    # Calculate theoretical profits
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    pi_n = profits_n[0]
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    pi_m = profits_m[0]
    
    delta_q = (avg_profit_q - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ppo = (avg_profit_ppo - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    return avg_price_q, avg_price_ppo, delta_q, delta_ppo, update_count

models = ['logit', 'hotelling', 'linear']
num_runs = 5
results = {}

for model in models:
    print(f"\n=== Running {model.upper()} model ===")
    
    # Get theoretical price
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    
    avg_prices_q = []
    avg_prices_ppo = []
    deltas_q = []
    deltas_ppo = []
    update_counts = []
    
    for run in range(num_runs):
        seed = SEED + run
        print(f"\nRun {run+1}/{num_runs} with seed {seed}")
        
        apq, app, dq, dp, updates = run_simulation(model, seed)
        avg_prices_q.append(apq)
        avg_prices_ppo.append(app)
        deltas_q.append(dq)
        deltas_ppo.append(dp)
        update_counts.append(updates)
        
        print(f"  Q-Learning: price={apq:.3f}, delta={dq:.3f}")
        print(f"  PPO: price={app:.3f}, delta={dp:.3f}, updates={updates}")
    
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price PPO': np.mean(avg_prices_ppo),
        'Delta Q': np.mean(deltas_q),
        'Delta PPO': np.mean(deltas_ppo),
        'Avg Updates': np.mean(update_counts)
    }

# Create DataFrame
data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'PPO Avg. Prices': [round(results[m]['Avg Price PPO'], 2) for m in models],
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 2) for m in models],
    'PPO Extra-profits Δ': [round(results[m]['Delta PPO'], 2) for m in models],
    'PPO Updates': [int(results[m]['Avg Updates']) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/q_vs_ppo.csv")
print("\n=== Final Results ===")
print(df)
