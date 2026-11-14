import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from puyopuyo import HEIGHT, PuyoEnv, WIDTH

COLOR_MAP = {None: 0, "R": 1, "G": 2, "B": 3, "Y": 4}

# Generate all 22 valid (target_x, target_rotation) placements
PLACEMENTS = []
for rot in range(4):
    for x in range(WIDTH):
        if rot == 1 and x == WIDTH - 1:  # Child extends right out of bounds
            continue
        if rot == 3 and x == 0:  # Child extends left out of bounds
            continue
        PLACEMENTS.append((x, rot))


class PuyoGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = PuyoEnv(animate=False)
        self.action_space = spaces.Discrete(len(PLACEMENTS))
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 1, shape=(5, HEIGHT, WIDTH), dtype=np.float32),
            "pieces": spaces.Box(0, 1, shape=(24,), dtype=np.float32),
        })

    def _encode_obs(self):
        obs = self.env.observe()
        board = np.zeros((5, HEIGHT, WIDTH), dtype=np.float32)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                board[COLOR_MAP[obs["board"][y][x]], y, x] = 1.0

        pieces = np.zeros(24, dtype=np.float32)
        all_pairs = []
        if obs["current"]:
            all_pairs.append(obs["current"]["colors"])
        all_pairs.extend(obs["next"][:2])

        for i, pair in enumerate(all_pairs[:3]):
            for j, color in enumerate(pair):
                idx = i * 8 + j * 4 + (COLOR_MAP[color] - 1)
                pieces[idx] = 1.0

        return {"board": board, "pieces": pieces}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        return self._encode_obs(), {}

    def step(self, action_idx):
        target_x, target_rot = PLACEMENTS[action_idx]

        # Rotate to target orientation
        while self.env.current and self.env.current["rotation"] != target_rot:
            if not self.env._rotate(1):
                break

        # Slide horizontally to target column
        while self.env.current and self.env.current["x"] != target_x:
            dx = 1 if target_x > self.env.current["x"] else -1
            if not self.env._move(dx, 0):
                break

        # Hard drop and resolve chains
        raw_reward, done, info = self.env.step("drop")

        # Shaped reward to encourage multi-chains and survival
        reward = (raw_reward / 10.0) + (2.0 * (info["chain"] ** 2))
        if done:
            reward -= 10.0

        return self._encode_obs(), reward, done, False, info


class PuyoCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 64 channels * 12 height * 6 width = 4608 + 24 piece features
        self.fc = nn.Sequential(
            nn.Linear(64 * HEIGHT * WIDTH + 24, 256),
            nn.ReLU(),
        )

    def forward(self, obs):
        board_feat = self.cnn(obs["board"])
        combined = th.cat([board_feat, obs["pieces"]], dim=1)
        return self.fc(combined)


if __name__ == "__main__":
    env = PuyoGymWrapper()
    policy_kwargs = dict(features_extractor_class=PuyoCNN)
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save("puyo_ppo_agent")
