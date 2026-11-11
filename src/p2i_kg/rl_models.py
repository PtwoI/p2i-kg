"""Small executable neural components for reinforcement-learning education.

These are deterministic, randomly initialized structural examples, not trained
agents or reproductions of the original benchmark results.
"""
import torch
from torch import nn


class DQNExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.q_head = nn.Linear(32, 4)

    def forward(self, observation):
        return self.q_head(self.encoder(observation))


class ActorCriticExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.Tanh())
        self.policy_head = nn.Linear(32, 4)
        self.value_head = nn.Linear(32, 1)

    def forward(self, observation):
        features = self.encoder(observation)
        return self.policy_head(features), self.value_head(features)


class DreamerExample(nn.Module):
    """One observation update followed by a two-step imagined rollout."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ELU())
        self.posterior = nn.Linear(16, 16)
        self.dynamics = nn.GRUCell(20, 16)
        self.actor = nn.Sequential(nn.Linear(16, 4), nn.Tanh())
        self.reward_head = nn.Linear(16, 1)
        self.value_head = nn.Linear(16, 1)
        self.decoder = nn.Linear(16, 8)

    def forward(self, observation):
        state = torch.tanh(self.posterior(self.encoder(observation)))
        reconstruction = self.decoder(state)
        rewards, values = [], []
        for _ in range(2):
            action = self.actor(state)
            state = self.dynamics(torch.cat((state, action), -1), state)
            rewards.append(self.reward_head(state))
            values.append(self.value_head(state))
        return reconstruction, torch.stack(rewards, 1), torch.stack(values, 1)


class DIAYNExample(nn.Module):
    """Skill-conditioned policy and state-to-skill classifier."""
    def __init__(self):
        super().__init__()
        self.skill_embedding = nn.Embedding(4, 4)
        self.policy = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 3))
        self.discriminator = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, state, skill):
        conditioned = torch.cat((state, self.skill_embedding(skill)), -1)
        return self.policy(conditioned), self.discriminator(state)


class SACExample(nn.Module):
    """Stochastic actor parameters and twin state-action critics."""
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 6))
        self.critic_one = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 1))
        self.critic_two = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, state):
        mean, log_std = self.actor(state).chunk(2, -1)
        action = torch.tanh(mean)
        state_action = torch.cat((state, action), -1)
        return mean, log_std, self.critic_one(state_action), self.critic_two(state_action)


class MuZeroExample(nn.Module):
    """Representation, one latent transition, reward, policy and value heads."""
    def __init__(self):
        super().__init__()
        self.representation = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.dynamics = nn.Sequential(nn.Linear(20, 16), nn.ReLU())
        self.reward_head = nn.Linear(16, 1)
        self.policy_head = nn.Linear(16, 4)
        self.value_head = nn.Linear(16, 1)

    def forward(self, observation, action):
        state = self.representation(observation)
        next_state = self.dynamics(torch.cat((state, action), -1))
        return self.reward_head(next_state), self.policy_head(next_state), self.value_head(next_state)


class DecisionTransformerExample(nn.Module):
    """Return, state and action tokens with causal self-attention."""
    def __init__(self):
        super().__init__()
        self.return_embedding = nn.Linear(1, 16)
        self.state_embedding = nn.Linear(8, 16)
        self.action_embedding = nn.Linear(4, 16)
        self.transformer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.action_head = nn.Linear(16, 4)

    def forward(self, returns, states, actions):
        tokens = torch.stack((self.return_embedding(returns), self.state_embedding(states),
                              self.action_embedding(actions)), dim=2).flatten(1, 2)
        length = tokens.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.transformer(tokens, src_mask=mask)
        return self.action_head(encoded[:, 1::3])


class DreamerDiscreteExample(nn.Module):
    """Compact categorical-latent world model with actor and critic."""
    def __init__(self, symlog=False):
        super().__init__()
        self.symlog = symlog
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ELU())
        self.posterior = nn.Linear(16, 16)
        self.dynamics = nn.GRUCell(20, 16)
        self.actor = nn.Sequential(nn.Linear(16, 4), nn.Tanh())
        self.reward_head = nn.Linear(16, 1)
        self.value_head = nn.Linear(16, 1)
        self.decoder = nn.Linear(16, 8)

    def forward(self, observation):
        if self.symlog:
            observation = observation.sign() * torch.log1p(observation.abs())
        logits = self.posterior(self.encoder(observation))
        state = torch.softmax(logits.reshape(-1, 4, 4), -1).flatten(1)
        reconstruction = self.decoder(state)
        action = self.actor(state)
        next_state = self.dynamics(torch.cat((state, action), -1), state)
        return reconstruction, self.reward_head(next_state), self.value_head(next_state)


class Dreamer4Example(nn.Module):
    """Action-conditioned causal transformer and actor–critic teaching path."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 16)
        self.action_embedding = nn.Linear(4, 16)
        self.world_transformer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.decoder = nn.Linear(16, 8)
        self.actor = nn.Linear(16, 4)
        self.value_head = nn.Linear(16, 1)

    def forward(self, observations, actions):
        tokens = self.encoder(observations) + self.action_embedding(actions)
        length = tokens.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        features = self.world_transformer(tokens, src_mask=mask)
        return self.decoder(features), self.actor(features), self.value_head(features)


class DADSExample(nn.Module):
    """Skill-conditioned policy and skill-conditioned transition prediction."""
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 3))
        self.skill_dynamics = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, state, skill):
        conditioned = torch.cat((state, skill), -1)
        return self.policy(conditioned), self.skill_dynamics(conditioned)


class APSExample(nn.Module):
    """Task-conditioned actor and successor-feature representation."""
    def __init__(self):
        super().__init__()
        self.state_features = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        self.actor = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 3))
        self.successor_features = nn.Sequential(nn.Linear(15, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, state, task):
        conditioned = torch.cat((state, task), -1)
        action = self.actor(conditioned)
        successor = self.successor_features(torch.cat((conditioned, action), -1))
        return self.state_features(state), action, successor


class OptionCriticExample(nn.Module):
    """Option values, option policies and termination probabilities."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.option_values = nn.Linear(16, 4)
        self.option_policies = nn.Linear(16, 12)
        self.option_termination = nn.Linear(16, 4)

    def forward(self, state):
        hidden = self.encoder(state)
        policies = self.option_policies(hidden).reshape(-1, 4, 3)
        return self.option_values(hidden), policies, torch.sigmoid(self.option_termination(hidden))


class PolicyGradientExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.policy_head = nn.Linear(32, 4)

    def forward(self, observation):
        return self.policy_head(self.encoder(observation))


class DuelingQExample(nn.Module):
    def __init__(self, atoms=1):
        super().__init__()
        self.atoms = atoms
        self.encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.value_stream = nn.Linear(32, atoms)
        self.advantage_stream = nn.Linear(32, 4 * atoms)

    def forward(self, observation):
        hidden = self.encoder(observation)
        value = self.value_stream(hidden).reshape(-1, 1, self.atoms)
        advantage = self.advantage_stream(hidden).reshape(-1, 4, self.atoms)
        logits = value + advantage - advantage.mean(1, keepdim=True)
        return logits.squeeze(-1) if self.atoms == 1 else logits


class DeterministicControlExample(nn.Module):
    def __init__(self, twin=False):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3), nn.Tanh())
        self.critic_one = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 1))
        self.critic_two = (nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 1))
                           if twin else None)

    def forward(self, observation):
        action = self.actor(observation)
        state_action = torch.cat((observation, action), -1)
        first = self.critic_one(state_action)
        if self.critic_two is not None:
            return action, first, self.critic_two(state_action)
        return action, first


class AlphaZeroExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.policy_head = nn.Linear(32, 4)
        self.value_head = nn.Sequential(nn.Linear(32, 1), nn.Tanh())

    def forward(self, board):
        hidden = self.trunk(board)
        return self.policy_head(hidden), self.value_head(hidden)


class PlaNetExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ELU())
        self.posterior = nn.Linear(16, 16)
        self.dynamics = nn.GRUCell(20, 16)
        self.reward_head = nn.Linear(16, 1)
        self.decoder = nn.Linear(16, 8)

    def forward(self, observation, action):
        state = torch.tanh(self.posterior(self.encoder(observation)))
        next_state = self.dynamics(torch.cat((state, action), -1), state)
        return self.decoder(next_state), self.reward_head(next_state)


class WorldModelsExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae_encoder = nn.Linear(8, 16)
        self.memory = nn.GRUCell(20, 16)
        self.controller = nn.Linear(32, 4)
        self.vae_decoder = nn.Linear(16, 8)

    def forward(self, observation, previous_action):
        latent = self.vae_encoder(observation)
        hidden = torch.zeros_like(latent)
        hidden = self.memory(torch.cat((latent, previous_action), -1), hidden)
        return self.vae_decoder(latent), self.controller(torch.cat((latent, hidden), -1))
