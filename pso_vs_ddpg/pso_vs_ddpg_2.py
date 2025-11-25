import sys
import os
import numpy as np
import pandas as pd

# Import the implementations
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from environments import MarketEnvContinuous
from agents import PSOAgent, DDPGAgent
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)

SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run a single simulation of PSO vs DDPG."""
    np.random.seed(seed)
    
    # Initialize environment
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    
    # Get bounds from env
    price_min = env.price_grid.min()
    price_max = env.price_grid.max()
    
    # Initialize agents with bounds
    pso_agent = PSOAgent(env, agent_id=0, price_min=price_min, price_max=price_max)
    
    # Initialize DDPG with proper parameters and bounds
    ddpg_agent = DDPGAgent(
        agent_id=1, 
        state_dim=2,  # MUST be 2 for pricing
        action_dim=1,
        seed=seed,
        price_min=price_min,
        price_max=price_max
    )
    
    # Reset environment
    shared_state = env.reset()
    
    # Track metrics
    profits_history = []
    prices_history = []
    
    state_pso = shared_state[0]  # PSO only needs opponent's price
    state_ddpg = shared_state.astype(np.float32)
    
    for t in range(env.horizon):
        # PSO updates with DDPG's last price
        pso_agent.update(shared_state[1])
        pso_price = pso_agent.choose_price()  # float
        
        # DDPG selects action
        ddpg_price, ddpg_norm_action = ddpg_agent.select_action(state_ddpg, explore=True)
        
        # Actions are continuous prices
        actions = [pso_price, ddpg_price]
        
        # Step environment
        shared_next_state, rewards, done, info = env.step(actions)
        
        next_state_pso = shared_next_state[0]
        next_state_ddpg = shared_next_state.astype(np.float32)
        
        # DDPG remembers and replays
        ddpg_agent.remember(state_ddpg, ddpg_norm_action, rewards[1], next_state_ddpg, done)
        ddpg_agent.replay()
        ddpg_agent.update_epsilon()
        
        # Update states
        state_pso = next_state_pso
        state_ddpg = next_state_ddpg
        
        # Record
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    # Compute averages over last 1000 steps
    last_prices = np.array(prices_history[-1000:])
    avg_price_pso = np.mean(last_prices[:, 0])
    avg_price_ddpg = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_pso = np.mean(last_profits[:, 0])
    avg_profit_ddpg = np.mean(last_profits[:, 1])
    
    # Benchmarks
    p_n = benchmarks['p_N']
    p_m = benchmarks['p_M']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    # Deltas
    delta_pso = (avg_profit_pso - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    delta_ddpg = (avg_profit_ddpg - pi_n) / (pi_m - pi_n) if (pi_m - pi_n) != 0 else 0
    
    # RPDIs
    rpdi_pso = (avg_price_pso - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    rpdi_ddpg = (avg_price_ddpg - p_n) / (p_m - p_n) if (p_m - p_n) != 0 else 0
    
    return avg_price_pso, avg_price_ddpg, delta_pso, delta_ddpg, rpdi_pso, rpdi_ddpg, p_n

models = ['logit', 'hotelling', 'linear']
num_runs = 1
shock_cfg = None
tb_calculator = TheoreticalBenchmarks()
all_benchmarks = tb_calculator.calculate_all_benchmarks(shock_cfg)
results = {}

# Store individual run results for logging
run_logs = {model: {'delta1': [], 'delta2': [], 'rpdi1': [], 'rpdi2': []} for model in models}

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.upper()}")
    print(f"{'='*60}")
    
    model_benchmarks = all_benchmarks[model]
    
    avg_prices_pso = []
    avg_prices_ddpg = []
    deltas_pso = []
    deltas_ddpg = []
    rpdis_pso = []
    rpdis_ddpg = []
    theo_prices = []
    
    for run in range(num_runs):
        seed = SEED + run
        apso, addpg, dpso, dddpg, rpso, rddpg, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
        avg_prices_pso.append(apso)
        avg_prices_ddpg.append(addpg)
        deltas_pso.append(dpso)
        deltas_ddpg.append(dddpg)
        rpdis_pso.append(rpso)
        rpdis_ddpg.append(rddpg)
        theo_prices.append(p_n)
        
        print(f"\nRun {run + 1}:")
        print(f"  PSO  -> Delta: {dpso:.4f}, RPDI: {rpso:.4f}")
        print(f"  DDPG -> Delta: {dddpg:.4f}, RPDI: {rddpg:.4f}")
        
        run_logs[model]['delta1'].append(dpso)
        run_logs[model]['delta2'].append(dddpg)
        run_logs[model]['rpdi1'].append(rpso)
        run_logs[model]['rpdi2'].append(rddpg)
    
    results[model] = {
        'Avg Price PSO': np.mean(avg_prices_pso),
        'Theo Price': np.mean(theo_prices),
        'Avg Price DDPG': np.mean(avg_prices_ddpg),
        'Delta PSO': np.mean(deltas_pso),
        'Delta DDPG': np.mean(deltas_ddpg),
        'RPDI PSO': np.mean(rpdis_pso),
        'RPDI DDPG': np.mean(rpdis_ddpg)
    }
    
    print(f"\n  Model Average: PSO Δ = {results[model]['Delta PSO']:.3f}, DDPG Δ = {results[model]['Delta DDPG']:.3f}")

print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}\n")

data = {
    'Model': [m.upper() for m in models],
    'PSO Avg. Prices': [round(results[m]['Avg Price PSO'], 2) for m in models],
    'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
    'DDPG Avg. Prices': [round(results[m]['Avg Price DDPG'], 2) for m in models],
    'Theo. Prices.1': [round(results[m]['Theo Price'], 2) for m in models],
    'PSO Extra-profits Δ': [round(results[m]['Delta PSO'], 2) for m in models],
    'DDPG Extra-profits Δ': [round(results[m]['Delta DDPG'], 2) for m in models],
    'PSO RPDI': [round(results[m]['RPDI PSO'], 2) for m in models],
    'DDPG RPDI': [round(results[m]['RPDI DDPG'], 2) for m in models]
}

df = pd.DataFrame(data)
df.to_csv("./results/pso_vs_ddpg_2.csv", index=False)
print("[Results saved to ./results/pso_vs_ddpg_2.csv]")
print(df.to_string(index=False))

print(f"\n{'='*80}")
print("OVERALL AVERAGES ACROSS ALL MODELS")
print(f"{'='*80}\n")

avg_delta1 = np.mean([results[m]['Delta PSO'] for m in models])
avg_delta2 = np.mean([results[m]['Delta DDPG'] for m in models])
avg_rpdi1 = np.mean([results[m]['RPDI PSO'] for m in models])
avg_rpdi2 = np.mean([results[m]['RPDI DDPG'] for m in models])

print("PSO:")
print(f"  Average Delta (Δ):  {avg_delta1:.4f}")
print(f"  Average RPDI:       {avg_rpdi1:.4f}")
print("\nDDPG:")
print(f"  Average Delta (Δ):  {avg_delta2:.4f}")
print(f"  Average RPDI:       {avg_rpdi2:.4f}")
