import os
import numpy as np
from agent_hybrid import BanditAgent

BASE = os.path.dirname(os.path.abspath(__file__))
RWD_DIR = os.path.join(BASE, "rwd_seq_examples")

def evaluate(agent, rewards_data):
    total_reward = 0
    for t in range(len(rewards_data)):
        chosen_arm = agent.select_arm()
        reward = rewards_data[t, chosen_arm]
        agent.update(chosen_arm, reward)
        total_reward += reward
    return total_reward

for fname in [
    "rwd_seq_example_01.npy",
    "rwd_seq_example_02.npy",
    "rwd_seq_example_03.npy",
]:
    rewards_data = np.load(os.path.join(RWD_DIR, fname))
    agent = BanditAgent(n_arms=2, seed=0)  # seed 고정 권장
    output = evaluate(agent, rewards_data)
    print(f"{fname} => Total # of rewards : {int(output)}")
