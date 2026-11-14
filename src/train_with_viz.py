import gymnasium as gym
import numpy as np
import pygame
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from puyo_env import HEIGHT, PuyoEnv, WIDTH

COLOR_MAP = {None: 0, "R": 1, "G": 2, "B": 3, "Y": 4}

PLACEMENTS = []
for rot in range(4):
    for x in range(WIDTH):
        if (rot == 1 and x == WIDTH - 1) or (rot == 3 and x == 0):
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

    def _board_potential(self):
        """Evaluate board quality: reward color clusters (2s & 3s) and penalize high stacks."""
        # 1. Reward connected groups of 2 or 3 matching colors ready for chains
        cluster_score = 0.0
        for group in self.env._components():
            size = len(group)
            if size == 2:
                cluster_score += 0.4
            elif size == 3:
                cluster_score += 1.2

        # 2. Penalize stack height, especially in the spawn column (x=2)
        height_penalty = 0.0
        for x in range(WIDTH):
            col_height = 0
            for y in range(HEIGHT):
                if self.env.board[y][x] is not None:
                    col_height = HEIGHT - y
                    break
            # Extra penalty for blocking the spawn chute (x=2)
            weight = 0.25 if x == 2 else (0.12 if x in (1, 3) else 0.06)
            if col_height >= 6:
                height_penalty += weight * (col_height - 5)

        return cluster_score - height_penalty

    def step(self, action_idx):
        prev_potential = self._board_potential()

        target_x, target_rot = PLACEMENTS[action_idx]
        while self.env.current and self.env.current["rotation"] != target_rot:
            if not self.env._rotate(1):
                break
        while self.env.current and self.env.current["x"] != target_x:
            dx = 1 if target_x > self.env.current["x"] else -1
            if not self.env._move(dx, 0):
                break

        raw_reward, done, info = self.env.step("drop")
        new_potential = self._board_potential()

        # Base score + exponential chain bonus + board improvement delta
        chain = info["chain"]
        chain_bonus = (3.5 * (chain ** 2.2)) if chain > 0 else 0.0
        shaping = new_potential - prev_potential

        reward = (raw_reward / 10.0) + chain_bonus + shaping
        if done:
            reward -= 15.0

        # Scale down so value_loss stays stable (< 20 instead of > 1,000)
        reward /= 10.0

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
        self.fc = nn.Sequential(
            nn.Linear(64 * HEIGHT * WIDTH + 24, 256),
            nn.ReLU(),
        )

    def forward(self, obs):
        board_feat = self.cnn(obs["board"])
        combined = th.cat([board_feat, obs["pieces"]], dim=1)
        return self.fc(combined)


class NeuralNetVisualizerCallback(BaseCallback):
    """Draws live neural network weights, activations, and board state in Pygame."""

    def __init__(self, render_every=500, width=1100, height=680):
        super().__init__()
        self.render_every = render_every
        self.width = width
        self.height = height
        self.activations = {}
        self.running = True

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Puyo Puyo PPO - Live Neural Network Visualizer")
        self.font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 18, bold=True)

    def _on_training_start(self):
        policy = self.model.policy
        policy.features_extractor.cnn.register_forward_hook(
            self._save_activation("cnn")
        )
        policy.mlp_extractor.policy_net.register_forward_hook(
            self._save_activation("hidden")
        )
        policy.action_net.register_forward_hook(
            self._save_activation("output")
        )

    def _save_activation(self, name):
        def hook(module, inp, out):
            data = out.detach().cpu().numpy()[0]
            if name == "cnn":
                data = data.reshape(64, HEIGHT, WIDTH).mean(axis=(1, 2))
            self.activations[name] = data

        return hook

    def _normalize(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-6:
            return np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    def _draw_board(self, raw_env):
        """Draw mini live Puyo board on the left side of the window."""
        cell = 22
        ox, oy = 30, 80
        colors = {
            None: (30, 33, 42),
            "R": (235, 75, 75),
            "G": (75, 215, 110),
            "B": (75, 145, 245),
            "Y": (245, 210, 65),
        }
        pygame.draw.rect(
            self.screen, (60, 65, 80),
            (ox - 4, oy - 4, WIDTH * cell + 8, HEIGHT * cell + 8), 2
        )
        for y in range(HEIGHT):
            for x in range(WIDTH):
                c = colors[raw_env.board[y][x]]
                rect = (ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2)
                pygame.draw.rect(self.screen, c, rect, border_radius=5)

        stats = [
            f"Steps: {self.num_timesteps:,}",
            f"Score: {raw_env.score}",
            f"Chain: {raw_env.chain}",
        ]
        for i, line in enumerate(stats):
            txt = self.font.render(line, True, (220, 225, 235))
            self.screen.blit(txt, (ox, oy + HEIGHT * cell + 20 + i * 22))

    def _draw_network(self, finished=False):
        self.screen.fill((18, 20, 26))

        status = "TRAINING COMPLETE - Close window to exit" if finished else "TRAINING LIVE..."
        header_color = (100, 240, 150) if finished else (120, 190, 255)
        header = self.title_font.render(f"PPO Policy Network ({status})", True, header_color)
        self.screen.blit(header, (30, 25))

        raw_env = self.training_env.envs[0].unwrapped.env
        self._draw_board(raw_env)

        if len(self.activations) < 3:
            pygame.display.flip()
            return

        cnn_act = self._normalize(self.activations["cnn"][:16])
        hid_act = self._normalize(self.activations["hidden"][:20])
        logits = self.activations["output"]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best_action = int(np.argmax(probs))

        weights = self.model.policy.action_net.weight.detach().cpu().numpy()[:, :20]
        w_max = np.abs(weights).max() + 1e-6

        layer_x = [310, 580, 860]
        layers = [cnn_act, hid_act, probs]
        labels = ["CNN Maps (16/64)", "Policy Hidden (20/64)", "Placements (22)"]

        positions = []
        for lx, vals, label in zip(layer_x, layers, labels):
            lbl_surf = self.font.render(label, True, (180, 185, 200))
            self.screen.blit(lbl_surf, (lx - 55, 55))
            n = len(vals)
            spacing = (self.height - 140) / max(n - 1, 1)
            coords = [(lx, int(95 + i * spacing)) for i in range(n)]
            positions.append(coords)

        # Draw connections: CNN -> Hidden
        for p1 in positions[0]:
            for p2 in positions[1]:
                pygame.draw.line(self.screen, (35, 40, 52), p1, p2, 1)

        # Draw weighted connections: Hidden -> Output
        for out_i, p2 in enumerate(positions[2]):
            for hid_i, p1 in enumerate(positions[1]):
                w = weights[out_i, hid_i] / w_max
                if abs(w) < 0.18:
                    continue
                intensity = int(min(255, abs(w) * 240))
                color = (40, intensity, 90) if w > 0 else (intensity, 55, 65)
                thickness = 2 if out_i == best_action else 1
                pygame.draw.line(self.screen, color, p1, p2, thickness)

        # Draw neuron nodes & action labels
        for l_idx, (coords, vals) in enumerate(zip(positions, layers)):
            for n_idx, ((nx, ny), val) in enumerate(zip(coords, vals)):
                if l_idx == 2:
                    is_best = (n_idx == best_action)
                    glow = int(min(255, 50 + val * 600))
                    fill = (80, 250, 140) if is_best else (50, glow, 220)
                    pygame.draw.circle(self.screen, fill, (nx, ny), 8)
                    pygame.draw.circle(self.screen, (230, 235, 245), (nx, ny), 8, 1)

                    tx, trot = PLACEMENTS[n_idx]
                    tag = f"x={tx} r={trot} ({probs[n_idx] * 100:4.1f}%)"
                    t_color = (120, 255, 160) if is_best else (170, 175, 190)
                    txt = self.font.render(tag, True, t_color)
                    self.screen.blit(txt, (nx + 16, ny - 7))
                else:
                    brightness = int(45 + val * 210)
                    fill = (brightness, int(brightness * 0.85), 255)
                    pygame.draw.circle(self.screen, fill, (nx, ny), 7)
                    pygame.draw.circle(self.screen, (140, 150, 170), (nx, ny), 7, 1)

        pygame.display.flip()

    def _on_step(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

        if self.num_timesteps % self.render_every == 0:
            self._draw_network(finished=False)
        return True

    def keep_open_until_closed(self):
        """Freezes the final network graph on screen until the user closes Pygame."""
        if not self.running:
            pygame.quit()
            return
        self._draw_network(finished=True)
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
            clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    env = PuyoGymWrapper()
    policy_kwargs = dict(features_extractor_class=PuyoCNN)
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        ent_coef=0.015,
        verbose=1,
    )

    viz_callback = NeuralNetVisualizerCallback(render_every=500)
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path="./checkpoints/",
        name_prefix="puyo_ppo",
    )
    callbacks = CallbackList([viz_callback, checkpoint_callback])

    model.learn(total_timesteps=2_000_000, callback=callbacks)
    model.save("puyo_ppo_agent")

    viz_callback.keep_open_until_closed()
