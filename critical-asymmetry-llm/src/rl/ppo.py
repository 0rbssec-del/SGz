import torch
import torch.nn.functional as F


class PPOTrainer:
    def __init__(self, policy, value_net, lr=3e-4, clip=0.2, K_epochs=4):
        self.policy = policy
        self.value_net = value_net
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=lr)
        self.clip = clip
        self.K_epochs = K_epochs

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        values = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, trajectories):
        for _ in range(self.K_epochs):
            for traj in trajectories:
                states = torch.stack(traj["states"])
                actions = torch.stack(traj["actions"])
                old_logp = torch.stack(traj["logps"])
                rewards = traj["rewards"]
                values = traj["values"]
                advantages, returns = self.compute_gae(rewards, values)
                advantages = torch.tensor(advantages, dtype=torch.float32)
                returns = torch.tensor(returns, dtype=torch.float32)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                logp, entropy = self.policy.evaluate(states, actions)
                ratio = (logp - old_logp).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                value_pred = self.value_net(states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, returns)
                loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
