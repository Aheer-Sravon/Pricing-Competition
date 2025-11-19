import sys
import os
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)  # Insert at beginning to prioritize it

from environments import MarketEnvContinuous
from agents import DQNAgent  # Assuming DQNAgent is added to agents or adjust import
from theoretical_benchmarks import TheoreticalBenchmarks

sys.path.pop(0)


SEED = 99

def run_simulation(model, seed, shock_cfg, benchmarks):
    """Run DQN vs DQN simulation"""
    np.random.seed(seed)
    
    env = MarketEnvContinuous(market_model=model, shock_cfg=shock_cfg, seed=seed)
    dqn_agent1 = DQNAgent(
        agent_id=0,
        state_dim=2,
        action_dim=env.N,
        loss_type='huber',
        use_double=True,
        seed=seed
    )
    dqn_agent2 = DQNAgent(
        agent_id=1,
        state_dim=2,
        action_dim=env.N,
        loss_type='huber',
        use_double=True,
        seed=seed
    )
    
    shared_state = env.reset()  # Returns shared (price_idx0, price_idx1)
    profits_history = []
    prices_history = []
    
    # NEW: Define agent-specific states (own, opp)
    state1 = (shared_state[0], shared_state[1])  # For Agent 1: (own, opp)
    state2 = (shared_state[1], shared_state[0])  # For Agent 2: (own, opp)
    
    for t in range(env.horizon):
        # CHANGED: Pass agent-specific state to each
        dqn_action1 = dqn_agent1.select_action(state1, explore=True)
        dqn_action2 = dqn_agent2.select_action(state2, explore=True)
        
        actions = [dqn_action1, dqn_action2]
        shared_next_state, rewards, done, info = env.step(actions)
        
        # NEW: Define agent-specific next_states
        next_state1 = (shared_next_state[0], shared_next_state[1])  # For Agent 1
        next_state2 = (shared_next_state[1], shared_next_state[0])  # For Agent 2
        
        # CHANGED: Use agent-specific state/next_state in remember
        dqn_agent1.remember(state1, dqn_action1, rewards[0], next_state1, done)
        dqn_agent1.replay()
        dqn_agent1.update_epsilon()
        
        dqn_agent2.remember(state2, dqn_action2, rewards[1], next_state2, done)
        dqn_agent2.replay()
        dqn_agent2.update_epsilon()
        
        # CHANGED: Update current states for next iteration
        state1 = next_state1
        state2 = next_state2
        shared_state = shared_next_state  # Optional, for consistency
        
        prices_history.append(info['prices'])
        profits_history.append(rewards)
    
    last_prices = np.array(prices_history[-1000:])
    avg_price_dqn1 = np.mean(last_prices[:, 0])
    avg_price_dqn2 = np.mean(last_prices[:, 1])
    
    last_profits = np.array(profits_history[-1000:])
    avg_profit_dqn1 = np.mean(last_profits[:, 0])
    avg_profit_dqn2 = np.mean(last_profits[:, 1])
    
    p_n = benchmarks['p_N']
    pi_n = benchmarks['E_pi_N']
    pi_m = benchmarks['E_pi_M']
    
    delta_dqn1 = (avg_profit_dqn1 - pi_n) / (pi_m - pi_n)
    delta_dqn2 = (avg_profit_dqn2 - pi_n) / (pi_m - pi_n)
    
    return avg_price_dqn1, avg_price_dqn2, delta_dqn1, delta_dqn2, p_n

def main():
    shock_cfg = {
        'enabled': False
    }
    
    benchmark_calculator = TheoreticalBenchmarks(seed=SEED)
    
    print("=" * 80)
    print("DQN vs DQN - NO SHOCKS")
    print("=" * 80)
    
    all_benchmarks = benchmark_calculator.calculate_all_benchmarks(shock_cfg)
    
    models = ['logit', 'hotelling', 'linear']
    num_runs = 5
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} simulations...")
        
        model_benchmarks = all_benchmarks[model]
        
        avg_prices_dqn1 = []
        avg_prices_dqn2 = []
        deltas_dqn1 = []
        deltas_dqn2 = []
        theo_prices = []
        
        for run in range(num_runs):
            seed = SEED + run
            apdqn1, apdqn2, ddqn1, ddqn2, p_n = run_simulation(model, seed, shock_cfg, model_benchmarks)
            avg_prices_dqn1.append(apdqn1)
            avg_prices_dqn2.append(apdqn2)
            deltas_dqn1.append(ddqn1)
            deltas_dqn2.append(ddqn2)
            theo_prices.append(p_n)
        
        results[model] = {
            'Avg Price DQN1': np.mean(avg_prices_dqn1),
            'Theo Price': np.mean(theo_prices),
            'Avg Price DQN2': np.mean(avg_prices_dqn2),
            'Delta DQN1': np.mean(deltas_dqn1),
            'Delta DQN2': np.mean(deltas_dqn2)
        }
        
        print(f"  Completed: DQN1 Δ = {results[model]['Delta DQN1']:.3f}, DQN2 Δ = {results[model]['Delta DQN2']:.3f}")
    
    data = {
        'Model': [m.upper() for m in models],
        'DQN1 Avg. Prices': [round(results[m]['Avg Price DQN1'], 2) for m in models],
        'Theo. Prices': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN2 Avg. Prices': [round(results[m]['Avg Price DQN2'], 2) for m in models],
        'Theo. Prices ': [round(results[m]['Theo Price'], 2) for m in models],
        'DQN1 Extra-profits Δ': [round(results[m]['Delta DQN1'], 2) for m in models],
        'DQN2 Extra-profits Δ': [round(results[m]['Delta DQN2'], 2) for m in models]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("./results/dqn_vs_dqn.csv", index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n[Results saved to ./results/dqn_vs_dqn.csv]")


if __name__ == "__main__":
    main()
