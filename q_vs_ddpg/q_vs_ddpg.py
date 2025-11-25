import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import QLearningAgent, DDPGAgent
sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, verbose=True):
    """Run a single simulation of Q-Learning vs DDPG."""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=seed)
    
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    q_agent = QLearningAgent(
        env.N,
        agent_id=0,
        price_grid=env.price_grid
    )
    
    ddpg_agent = DDPGAgent(
        agent_id=1,
        state_dim=2,
        action_dim=1,
        seed=seed,
        price_min=price_min,
        price_max=price_max
    )
    
    state = env.reset()
    
    profits_history = []
    prices_history = []
    
    if verbose:
        print(f"\nRunning {model.upper()} model simulation...")
        print(f"Horizon: {env.horizon} steps")
        print(f"Price grid size: {env.N}")
        print(f"Nash price: {env.P_N:.3f}")
        print(f"Monopoly price: {env.P_M:.3f}")
    
    for t in range(env.horizon):

        q_action_idx = q_agent.choose_action(state)
        q_price = env.price_grid[q_action_idx]
        
        ddpg_state = state.astype(np.float32)
        ddpg_price, ddpg_norm = ddpg_agent.select_action(ddpg_state, explore=True)
        
        actions = [q_price, ddpg_price]
        next_state, rewards, done, info = env.step(actions)
        
        q_agent.update(state, q_action_idx, rewards[0], next_state)
        
        next_ddpg_state = next_state.astype(np.float32)
        ddpg_agent.remember(ddpg_state, ddpg_norm, rewards[1], next_ddpg_state, done)
        ddpg_agent.replay()
        
        if t % 100 == 0:
            ddpg_agent.update_epsilon()
        
        state = next_state
        prices_history.append(info['prices'])
        profits_history.append(rewards)
        
        if verbose and (t + 1) % 1000 == 0:
            recent_prices = np.array(prices_history[-100:])
            avg_p_q = np.mean(recent_prices[:, 0])
            avg_p_ddpg = np.mean(recent_prices[:, 1])
            print(f"  Step {t+1}/{env.horizon}: Q price={avg_p_q:.3f}, DDPG price={avg_p_ddpg:.3f}, ε={ddpg_agent.epsilon:.3f}")
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_q = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_q = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    prices_n = np.array([env.P_N] * 2)
    _, profits_n = env.calculate_demand_and_profit(prices_n, np.zeros(2))
    
    prices_m = np.array([env.P_M] * 2)
    _, profits_m = env.calculate_demand_and_profit(prices_m, np.zeros(2))
    
    delta_q = (avg_profit_q - profits_n[0]) / (profits_m[0] - profits_n[0]) if (profits_m[0] - profits_n[0]) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - profits_n[1]) / (profits_m[1] - profits_n[1]) if (profits_m[1] - profits_n[1]) != 0 else 0
    
    rpdi_q = (avg_price_q - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - env.P_N) / (env.P_M - env.P_N) if (env.P_M - env.P_N) != 0 else 0
    
    return avg_price_q, avg_price_ddpg, delta_q, delta_ddpg, rpdi_q, rpdi_ddpg


models = ['logit', 'hotelling', 'linear']
num_runs = 1
results = {}

run_logs = {model: {'delta_q': [], 'delta_ddpg': [], 'rpdi_q': [], 'rpdi_ddpg': []} for model in models}

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    env_temp = MarketEnvContinuous(market_model=model, shock_cfg=None, seed=SEED)
    p_n = env_temp.P_N
    p_m = env_temp.P_M
    
    print(f"Nash equilibrium price: {p_n:.3f}")
    print(f"Monopoly price: {p_m:.3f}")
    
    avg_prices_q = []
    avg_prices_ddpg = []
    deltas_q = []
    deltas_ddpg = []
    rpdis_q = []
    rpdis_ddpg = []
    
    for run in range(num_runs):
        seed = SEED + run
        print(f"\nRun {run + 1}/{num_runs} (seed={seed})")
        
        apq, apd, dq, dd, rq, rd = run_simulation(model, seed, verbose=(run == 0))
        
        avg_prices_q.append(apq)
        avg_prices_ddpg.append(apd)
        deltas_q.append(dq)
        deltas_ddpg.append(dd)
        rpdis_q.append(rq)
        rpdis_ddpg.append(rd)
        
        print(f"\nResults for Run {run + 1}:")
        print(f"  Q-Learning  -> Price: {apq:.3f}, Delta: {dq:.4f}, RPDI: {rq:.4f}")
        print(f"  DDPG        -> Price: {apd:.3f}, Delta: {dd:.4f}, RPDI: {rd:.4f}")
        
        run_logs[model]['delta_q'].append(dq)
        run_logs[model]['delta_ddpg'].append(dd)
        run_logs[model]['rpdi_q'].append(rq)
        run_logs[model]['rpdi_ddpg'].append(rd)
    
    results[model] = {
        'Avg Price Q': np.mean(avg_prices_q),
        'Std Price Q': np.std(avg_prices_q),
        'Theo Price': p_n,
        'Avg Price DDPG': np.mean(avg_prices_ddpg),
        'Std Price DDPG': np.std(avg_prices_ddpg),
        'Delta Q': np.mean(deltas_q),
        'Std Delta Q': np.std(deltas_q),
        'Delta DDPG': np.mean(deltas_ddpg),
        'Std Delta DDPG': np.std(deltas_ddpg),
        'RPDI Q': np.mean(rpdis_q),
        'Std RPDI Q': np.std(rpdis_q),
        'RPDI DDPG': np.mean(rpdis_ddpg),
        'Std RPDI DDPG': np.std(rpdis_ddpg)
    }

print(f"\n{'='*80}")
print("SUMMARY RESULTS")
print(f"{'='*80}\n")

summary_data = {
    'Model': [],
    'Q-Learning Avg Price': [],
    'Q-Learning Delta': [],
    'Q-Learning RPDI': [],
    'DDPG Avg Price': [],
    'DDPG Delta': [],
    'DDPG RPDI': [],
    'Nash Price': []
}

for model in models:
    r = results[model]
    summary_data['Model'].append(model.upper())
    summary_data['Q-Learning Avg Price'].append(f"{r['Avg Price Q']:.3f}")
    summary_data['Q-Learning Delta'].append(f"{r['Delta Q']:.3f}")
    summary_data['Q-Learning RPDI'].append(f"{r['RPDI Q']:.3f}")
    summary_data['DDPG Avg Price'].append(f"{r['Avg Price DDPG']:.3f}")
    summary_data['DDPG Delta'].append(f"{r['Delta DDPG']:.3f}")
    summary_data['DDPG RPDI'].append(f"{r['RPDI DDPG']:.3f}")
    summary_data['Nash Price'].append(f"{r['Theo Price']:.3f}")

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

detailed_data = {
    'Model': [m.upper() for m in models],
    'Q Avg. Prices': [round(results[m]['Avg Price Q'], 3) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 3) for m in models],
    'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 3) for m in models],
    'Q Extra-profits Δ': [round(results[m]['Delta Q'], 3) for m in models],
    'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 3) for m in models],
    'Q RPDI': [round(results[m]['RPDI Q'], 3) for m in models],
    'DDPG RPDI': [round(results[m]['RPDI DDPG'], 3) for m in models]
}

df_detailed = pd.DataFrame(detailed_data)
df_detailed.to_csv("results/q_vs_ddpg_2.csv", index=False)
print("\nResults saved to q_vs_ddpg_2.csv")

print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta_q = np.mean([results[m]['Delta Q'] for m in models])
avg_delta_ddpg = np.mean([results[m]['Delta DDPG'] for m in models])
avg_rpdi_q = np.mean([results[m]['RPDI Q'] for m in models])
avg_rpdi_ddpg = np.mean([results[m]['RPDI DDPG'] for m in models])

print("Q-Learning:")
print(f"  Average Delta (Δ): {avg_delta_q:.4f}")
print(f"  Average RPDI:      {avg_rpdi_q:.4f}")
print("\nDDPG:")
print(f"  Average Delta (Δ): {avg_delta_ddpg:.4f}")
print(f"  Average RPDI:      {avg_rpdi_ddpg:.4f}")
