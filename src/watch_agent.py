import copy
import operator

import gymnasium as gym
import numpy as np
import pygame
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
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

    def step(self, action_idx):
        target_x, target_rot = PLACEMENTS[action_idx]
        while self.env.current and self.env.current["rotation"] != target_rot:
            if not self.env._rotate(1):
                break
        while self.env.current and self.env.current["x"] != target_x:
            dx = 1 if target_x > self.env.current["x"] else -1
            if not self.env._move(dx, 0):
                break

        raw_reward, done, info = self.env.step("drop")
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
        self.fc = nn.Sequential(
            nn.Linear(64 * HEIGHT * WIDTH + 24, 256),
            nn.ReLU(),
        )

    def forward(self, obs):
        board_feat = self.cnn(obs["board"])
        combined = th.cat([board_feat, obs["pieces"]], dim=1)
        return self.fc(combined)


def _simulate_placement(env_instance, action_idx):
    """Clone env, execute one macro-placement, and return (cloned_env, raw_reward, chain, done)."""
    sim = copy.deepcopy(env_instance)
    target_x, target_rot = PLACEMENTS[action_idx]

    while sim.current and sim.current["rotation"] != target_rot:
        if not sim._rotate(1):
            break
    while sim.current and sim.current["x"] != target_x:
        dx = 1 if target_x > sim.current["x"] else -1
        if not sim._move(dx, 0):
            break

    raw_reward, done, info = sim.step("drop")
    return sim, raw_reward, info["chain"], done


def _normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


class AgentPlayerVisualizer:
    def __init__(self, model_path="puyo_ppo_agent", width=1120, height=700):
        self.width = width
        self.height = height
        self.activations = {}
        self.env = PuyoGymWrapper()
        self.model = PPO.load(model_path, device="cpu")
        self._attach_hooks()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Puyo Puyo PPO - Trained Agent Playback")
        self.font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 18, bold=True)

        self.paused = False
        self.fps = 20
        self.games_played = 0
        self.high_score = 0
        self.completed_scores = []
        self.last_action = None

    def _evaluate_board(self, sim_env, raw_reward, chain, done):
        """Evaluate board survival, accessible chain triggers, and PPO Critic value."""
        if done:
            return -1e6

        board = sim_env.board

        # 1. Compute column heights
        col_heights = []
        for x in range(WIDTH):
            h = 0
            for y in range(HEIGHT):
                if board[y][x] is not None:
                    h = HEIGHT - y
                    break
            col_heights.append(h)

        h0, h1, h2, h3, h4, h5 = col_heights
        max_h = max(col_heights)

        # Heavy penalty for stacking near the spawn chute (x=2, y=0..1)
        spawn_danger = (h2 ** 2.4) * 2.5 + (h1 ** 1.9) * 0.8 + (h3 ** 1.9) * 0.8
        if h2 >= 8:
            spawn_danger += 800.0

        # Reward U-shaped basin: outer columns (0, 5) slightly higher than centre (2, 3)
        basin_penalty = (
                                abs((h0 - h2) - 2)
                                + abs((h1 - h2) - 1)
                                + abs((h3 - h2) - 0)
                                + abs((h4 - h3) - 1)
                                + abs((h5 - h3) - 2)
                        ) * 1.5

        # 2. Check connected clusters AND whether they touch empty air (reachable triggers)
        cluster_bonus = 0.0
        for group in sim_env._components():
            sz = len(group)
            if sz < 2:
                continue
            exposed = False
            for gx, gy in group:
                for nx, ny in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and board[ny][nx] is None:
                        exposed = True
                        break
                if exposed:
                    break

            if sz == 2:
                cluster_bonus += 3.5 if exposed else 1.0
            elif sz == 3:
                cluster_bonus += 14.0 if exposed else 2.0

        # 3. Chain scoring: huge reward for 2+ chains; hold 1-chains until board gets high
        if chain >= 2:
            chain_score = (raw_reward / 4.0) + (40.0 * (chain ** 2.6))
        elif chain == 1:
            chain_score = 18.0 if max_h >= 7 else -5.0
        else:
            chain_score = 0.0

        # 4. Trained PPO Critic value estimate
        temp_wrapper = PuyoGymWrapper()
        temp_wrapper.env = sim_env
        obs_dict = temp_wrapper._encode_obs()
        obs_tensor = {
            k: th.as_tensor(v, dtype=th.float32).unsqueeze(0)
            for k, v in obs_dict.items()
        }
        with th.no_grad():
            critic_val = float(self.model.policy.predict_values(obs_tensor).item())

        return chain_score + cluster_bonus + (critic_val * 1.5) - spawn_danger - basin_penalty

    def _choose_best_move_lookahead(self, beam_width=8):
        """2-ply beam search using CURRENT + NEXT visible pairs."""
        first_ply = []
        for a1 in range(len(PLACEMENTS)):
            sim1, r1, c1, d1 = _simulate_placement(self.env.env, a1)
            if d1:
                continue
            score1 = self._evaluate_board(sim1, r1, c1, d1)
            first_ply.append((score1, a1, sim1))

        if not first_ply:
            obs = self.env._encode_obs()
            act, _ = self.model.predict(obs, deterministic=True)
            return int(act)

        first_ply.sort(key=operator.itemgetter(0), reverse=True)
        best_total = -1e9
        _, best_action, _ = first_ply[0]

        for score1, a1, sim1 in first_ply[:beam_width]:
            best_ply2 = -1e9
            for a2 in range(len(PLACEMENTS)):
                sim2, r2, c2, d2 = _simulate_placement(sim1, a2)
                if d2:
                    continue
                score2 = self._evaluate_board(sim2, r2, c2, d2)
                if score2 > best_ply2:
                    best_ply2 = score2

            total_val = score1 + (0.95 * best_ply2)
            if total_val > best_total:
                best_total = total_val
                best_action = a1

        return best_action

    def _attach_hooks(self):
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
            batch_first = out[0].detach().cpu().numpy()
            if name == "cnn":
                batch_first = batch_first.reshape(64, HEIGHT, WIDTH).mean(axis=(1, 2))
            self.activations[name] = batch_first

        return hook

    def _draw_puyo(self, cx, cy, radius, color_code):
        colors = {
            "R": (235, 75, 75),
            "G": (75, 215, 110),
            "B": (75, 145, 245),
            "Y": (245, 210, 65),
        }
        if color_code in colors:
            pygame.draw.circle(self.screen, colors[color_code], (cx, cy), radius)
            pygame.draw.circle(
                self.screen, (255, 255, 255), (cx - 3, cy - 3), max(2, radius // 4)
            )

    def _draw_board_and_queue(self):
        raw_env = self.env.env
        cell = 24
        ox, oy = 30, 85

        pygame.draw.rect(
            self.screen,
            (70, 75, 92),
            (ox - 4, oy - 4, WIDTH * cell + 8, HEIGHT * cell + 8),
            2,
        )
        for y in range(HEIGHT):
            for x in range(WIDTH):
                rect = (ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2)
                pygame.draw.rect(self.screen, (28, 31, 40), rect, border_radius=4)
                piece = raw_env.board[y][x]
                if piece is not None:
                    self._draw_puyo(
                        ox + x * cell + cell // 2,
                        oy + y * cell + cell // 2,
                        cell // 2 - 2,
                        piece,
                    )

        qx = ox + WIDTH * cell + 22
        lbl = self.font.render("CURRENT", True, (180, 185, 200))
        self.screen.blit(lbl, (qx, oy))
        if raw_env.current:
            c0, c1 = raw_env.current["colors"]
            self._draw_puyo(qx + 18, oy + 32, 10, c1)
            self._draw_puyo(qx + 18, oy + 56, 10, c0)

        lbl_next = self.font.render("NEXT", True, (180, 185, 200))
        self.screen.blit(lbl_next, (qx, oy + 90))
        for idx, pair in enumerate(list(raw_env.pair_queue)[:2]):
            p0, p1 = pair
            base_y = oy + 122 + idx * 65
            self._draw_puyo(qx + 18, base_y, 9, p1)
            self._draw_puyo(qx + 18, base_y + 22, 9, p0)

        avg_score = int(np.mean(self.completed_scores)) if self.completed_scores else 0
        low_score = np.min(self.completed_scores) if self.completed_scores else 0

        stats = [
            f"Score:      {raw_env.score}",
            f"Avg Score:  {avg_score}",
            f"High Score: {self.high_score}",
            f"Low Score:  {low_score}",
            f"Last Chain: {raw_env.chain}",
            f"Games:      {self.games_played}",
            f"Speed:      {self.fps} moves/sec",
            "",
            "[SPACE] Pause/Resume",
            "[RIGHT] Step 1 move",
            "[UP/DN] Change speed",
            "[R]     Reset board",
        ]
        for i, line in enumerate(stats):
            color = (140, 220, 170) if i < 4 else (150, 155, 170)
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (ox, oy + HEIGHT * cell + 20 + i * 21))

    def _draw_network(self):
        self.screen.fill((18, 20, 26))

        state_tag = "PAUSED" if self.paused else f"RUNNING ({self.fps} FPS)"
        header_color = (255, 200, 90) if self.paused else (100, 240, 150)
        header = self.title_font.render(
            f"Trained PPO Agent - Live Decision Graph [{state_tag}]", True, header_color
        )
        self.screen.blit(header, (30, 25))

        self._draw_board_and_queue()

        if len(self.activations) < 3:
            pygame.display.flip()
            return

        cnn_act = _normalize(self.activations["cnn"][:16])
        hid_act = _normalize(self.activations["hidden"][:20])
        logits = self.activations["output"]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best_action = int(np.argmax(probs))

        weights = self.model.policy.action_net.weight.detach().cpu().numpy()[:, :20]
        w_max = np.abs(weights).max() + 1e-6

        layer_x = (380, 620, 860)
        layers = (cnn_act, hid_act, probs)
        labels = ("CNN Maps (16/64)", "Policy Hidden (20/64)", "Placements (22)")

        positions = []
        for lx, vals, label in zip(layer_x, layers, labels):
            lbl_surf = self.font.render(label, True, (180, 185, 200))
            self.screen.blit(lbl_surf, (lx - 60, 58))
            n = len(vals)
            spacing = (self.height - 130) / max(n - 1, 1)
            coords = [(lx, int(92 + i * spacing)) for i in range(n)]
            positions.append(coords)

        cnn_coords, hid_coords, out_coords = positions

        # Draw connections: CNN -> Hidden
        for p1 in cnn_coords:
            for p2 in hid_coords:
                pygame.draw.line(self.screen, (35, 40, 52), p1, p2, 1)

        # Draw weighted connections: Hidden -> Output
        for out_i, p2 in enumerate(out_coords):
            for hid_i, p1 in enumerate(hid_coords):
                w = weights[out_i, hid_i] / w_max
                if abs(w) < 0.18:
                    continue
                intensity = int(min(255, abs(w) * 240))
                color = (40, intensity, 90) if w > 0 else (intensity, 55, 65)
                thickness = 2 if out_i == best_action else 1
                pygame.draw.line(self.screen, color, p1, p2, thickness)

        # Draw neuron nodes & placement labels
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

    def run(self):
        obs, _ = self.env.reset()
        clock = pygame.time.Clock()
        running = True

        self.last_action = self._choose_best_move_lookahead()

        while running:
            step_once = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT and self.paused:
                        step_once = True
                    elif event.key == pygame.K_UP:
                        self.fps = min(20, self.fps + 1)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 1)
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        action, _ = self.model.predict(obs, deterministic=True)
                        self.last_action = int(action)

            if not self.paused or step_once:
                obs, _, done, _, info = self.env.step(self.last_action)
                self.high_score = max(self.high_score, info["score"])
                if done:
                    self.games_played += 1
                    self.completed_scores.append(info["score"])
                    obs, _ = self.env.reset()

                action, _ = self.model.predict(obs, deterministic=True)
                self.last_action = int(action)

            self._draw_network()
            clock.tick(self.fps if not self.paused else 30)

        pygame.quit()


if __name__ == "__main__":
    viewer = AgentPlayerVisualizer("puyo_ppo_agent")
    viewer.run()
