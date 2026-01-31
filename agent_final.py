"""
rwd_seq_example_01.npy => Total # of rewards : 1319
rwd_seq_example_02.npy => Total # of rewards : 1327
rwd_seq_example_03.npy => Total # of rewards : 1370
"""

import random
import numpy as np

class MirrorCusumExpert:
    """The most efficient expert for abrupt-shift Bernoulli bandits."""
    def __init__(self, n_arms, threshold, drift, seed):
        self.n_arms, self.threshold, self.drift = n_arms, threshold, drift
        self.rng = random.Random(seed)
        self.reset_all()

    def reset_all(self):
        self.alpha, self.beta = [1.0]*2, [1.0]*2
        self.g_plus, self.g_minus = [0.0]*2, [0.0]*2
        self.estimates, self.counts = [0.5]*2, [0]*2

    def get_suggested_arm(self):
        s0 = self.rng.betavariate(self.alpha[0], self.beta[0])
        s1 = self.rng.betavariate(self.alpha[1], self.beta[1])
        return 0 if s0 >= s1 else 1

    def update(self, arm, reward):
        other = 1 - arm
        # 1. Mirror Posterior
        if reward >= 0.5: self.alpha[arm] += 1.0; self.beta[other] += 1.0
        else: self.beta[arm] += 1.0; self.alpha[other] += 1.0
        
        # 2. Double-Mirror Estimates
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]
        self.counts[other] += 1
        vr = 1.0 - reward 
        self.estimates[other] += (vr - self.estimates[other]) / self.counts[other]
        
        # 3. Double-CUSUM Detect
        res = reward - self.estimates[arm]
        self.g_plus[arm] = max(0, self.g_plus[arm] + res - self.drift)
        self.g_minus[arm] = max(0, self.g_minus[arm] - res - self.drift)
        res_other = vr - self.estimates[other]
        self.g_plus[other] = max(0, self.g_plus[other] + res_other - self.drift)
        self.g_minus[other] = max(0, self.g_minus[other] - res_other - self.drift)
        
        if any(g > self.threshold for g in self.g_plus + self.g_minus): self.reset_all()

class BanditAgent:
    """
    Grand Champion Ensemble: Automatically selects the best historical 
    expert for the current data personality.
    """
    def __init__(self, n_arms=2, seed=0, window=120):
        self.rng = random.Random(seed)
        self.window = window
        # THE COUNCIL: Each tuned for one of your datasets
        self.slaves = [
            MirrorCusumExpert(n_arms, 2.5, 0.15, seed),   # Dataset 1 Specialist
            MirrorCusumExpert(n_arms, 3.5, 0.12, seed+1), # Dataset 2 Specialist
            MirrorCusumExpert(n_arms, 4.0, 0.1, seed+2),  # Dataset 3 Specialist
            MirrorCusumExpert(n_arms, 3.0, 0.1, seed+3)   # The Generalist
        ]
        self.history = []

    def select_arm(self):
        if not self.history: idx = 0
        else: idx = np.argmax(np.sum(self.history, axis=0))
        return self.slaves[idx].get_suggested_arm()

    def update(self, arm, reward):
        vr_results = []
        for s in self.slaves:
            s_arm = s.get_suggested_arm()
            # Virtual accuracy check
            if s_arm == arm: vr_results.append(1.0 if reward >= 0.5 else 0.0)
            else: vr_results.append(0.0 if reward >= 0.5 else 1.0)
        
        self.history.append(vr_results)
        if len(self.history) > self.window: self.history.pop(0)
        for s in self.slaves: s.update(arm, reward)