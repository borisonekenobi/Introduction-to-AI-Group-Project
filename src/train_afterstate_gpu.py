import copy
import random
from collections import deque
import numpy as np
import pygame
import torch as th
import torch.nn as nn
import torch.optim as optim
from puyo_env import PuyoEnv, WIDTH, HEIGHT

COLOR_MAP = {None: 0, "R": 1, "G": 2, "B": 3, "Y": 4}

PLACEMENTS = []
for rot in range(4):
    for x in range(WIDTH):
        if (rot == 1 and x == WIDTH - 1) or (rot == 3 and x == 0):
            continue
        PLACEMENTS.append((x, rot))


def encode_afterstate(env_obj):
    """Encode board (5x12x6) + engineered structural features + next queue (34 floats)."""
    board = np.zeros((5, HEIGHT, WIDTH), dtype=np.float32)
    heights = np.zeros(WIDTH, dtype=np.float32)

    for x in range(WIDTH):
        for y in range(HEIGHT):
            cell = env_obj.board[y][x]
            board[COLOR_MAP[cell], y, x] = 1.0
            if cell is not None and heights[x] == 0:
                heights[x] = (HEIGHT - y) / float(HEIGHT)

    # Queue features (next 2 pairs = 16 floats)
    queue_vec = np.zeros(16, dtype=np.float32)
    for i, pair in enumerate(list(env_obj.pair_queue)[:2]):
        c0, c1 = pair
        queue_vec[i * 8 + (COLOR_MAP[c0] - 1)] = 1.0
        queue_vec[i * 8 + 4 + (COLOR_MAP[c1] - 1)] = 1.0

    # Structural summary features (6 heights + 6 height diffs + 6 cluster counts = 18 floats)
    diffs = np.abs(np.diff(heights))
    max_h = np.array([heights.max()], dtype=np.float32)

    c2, c3, c4 = 0.0, 0.0, 0.0
    for grp in env_obj._components():
        sz = len(grp)
        if sz == 2:
            c2 += 0.1
        elif sz == 3:
            c3 += 0.2
        elif sz >= 4:
            c4 += 0.3

    stats_vec = np.concatenate([heights, diffs, max_h, np.array([c2, c3, c4], dtype=np.float32)])
    meta = np.concatenate([queue_vec, stats_vec])  # 16 + 15 = 31 floats
    return board, meta


def get_valid_afterstates(base_env):
    """Simulate every unique valid placement from spawn and return resulting afterstates."""
    seen_boards = set()
    candidates = []

    for action_idx, (target_x, target_rot) in enumerate(PLACEMENTS):
        sim = copy.deepcopy(base_env)
        while sim.current and sim.current["rotation"] != target_rot:
            if not sim._rotate(1):
                break
        while sim.current and sim.current["x"] != target_x:
            dx = 1 if target_x > sim.current["x"] else -1
            if not sim._move(dx, 0):
                break

        raw_reward, done, info = sim.step("drop")

        # Deduplicate identical resulting board states
        board_sig = tuple(tuple(row) for row in sim.board)
        if board_sig in seen_boards and not done:
            continue
        seen_boards.add(board_sig)

        # Shaped immediate reward: huge bonus for multi-chains + survival step
        chain = info["chain"]
        if done:
            step_reward = -10.0
        else:
            chain_bonus = (2.5 * (chain ** 2.3)) if chain > 0 else 0.0
            step_reward = 0.15 + (raw_reward / 40.0) + chain_bonus

        b_arr, m_arr = encode_afterstate(sim)
        candidates.append((action_idx, sim, step_reward, done, info["score"], chain, b_arr, m_arr))

    return candidates


class AfterstateValueNet(nn.Module):
    """ResNet-inspired CNN + MLP Value Network V(s') for GPU batch evaluation."""

    def __init__(self, meta_dim=31):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(64 * HEIGHT * WIDTH + meta_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )

    def forward(self, board, meta):
        x = self.conv_in(board)
        x = self.relu(x + self.res_block(x))
        x = x.flatten(start_dim=1)
        x = th.cat([x, meta], dim=1)
        return self.head(x).squeeze(-1)


class GPUAfterstateTrainer:
    def __init__(self, total_episodes=15_000, render_every=15):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(f"Training Afterstate Value Network on: {self.device}")

        self.online_net = AfterstateValueNet().to(self.device)
        self.target_net = AfterstateValueNet().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.online_net.parameters(), lr=2.5e-4, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = deque(maxlen=120_000)
        self.batch_size = 512
        self.gamma = 0.985
        self.total_episodes = total_episodes
        self.render_every = render_every

        self.scores_history = deque(maxlen=100)
        self.high_score = 0
        self.best_avg = 0.0
        self.last_values = None
        self.last_chosen = 0

        # Pygame live visualizer
        pygame.init()
        self.width, self.height = 1100, 680
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RTX 2070 SUPER - Afterstate Value Network Training")
        self.font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.running = True

    def _select_action(self, candidates, epsilon):
        boards = th.as_tensor(np.stack([c[6] for c in candidates]), device=self.device)
        metas = th.as_tensor(np.stack([c[7] for c in candidates]), device=self.device)
        rewards = np.array([c[2] for c in candidates], dtype=np.float32)
        dones = np.array([c[3] for c in candidates], dtype=np.float32)

        with th.no_grad():
            future_vals = self.online_net(boards, metas).cpu().numpy()

        total_q = rewards + (1.0 - dones) * self.gamma * future_vals
        self.last_values = total_q

        if random.random() < epsilon:
            idx = random.randrange(len(candidates))
        else:
            idx = int(np.argmax(total_q))

        self.last_chosen = idx
        return candidates[idx]

    def _train_step(self):
        if len(self.replay_buffer) < 4096:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        b_curr = th.as_tensor(np.stack([x[0] for x in batch]), device=self.device)
        m_curr = th.as_tensor(np.stack([x[1] for x in batch]), device=self.device)
        rewards = th.as_tensor([x[2] for x in batch], dtype=th.float32, device=self.device)
        b_next = th.as_tensor(np.stack([x[3] for x in batch]), device=self.device)
        m_next = th.as_tensor(np.stack([x[4] for x in batch]), device=self.device)
        dones = th.as_tensor([x[5] for x in batch], dtype=th.float32, device=self.device)

        pred_v = self.online_net(b_curr, m_curr)
        with th.no_grad():
            next_v = self.target_net(b_next, m_next)
            target_v = rewards + (1.0 - dones) * self.gamma * next_v

        loss = self.loss_fn(pred_v, target_v)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 2.0)
        self.optimizer.step()

    def _draw_viz(self, env_obj, episode, epsilon, finished=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.screen.fill((18, 20, 26))
        status = "TRAINING COMPLETE - Close window to exit" if finished else f"GPU TRAINING ({self.device})"
        header = self.title_font.render(f"Afterstate Value Network - {status}", True, (100, 240, 150))
        self.screen.blit(header, (30, 22))

        # Draw Puyo board
        cell = 23
        ox, oy = 30, 80
        colors = {
            None: (28, 31, 40),
            "R": (235, 75, 75),
            "G": (75, 215, 110),
            "B": (75, 145, 245),
            "Y": (245, 210, 65),
        }
        pygame.draw.rect(self.screen, (65, 70, 88), (ox - 4, oy - 4, WIDTH * cell + 8, HEIGHT * cell + 8), 2)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                c = colors[env_obj.board[y][x]]
                pygame.draw.rect(self.screen, c, (ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2), border_radius=5)

        avg_100 = np.mean(self.scores_history) if self.scores_history else 0.0
        stats = [
            f"Episode:      {episode} / {self.total_episodes}",
            f"100-Game Avg: {avg_100:.1f}",
            f"Best 100-Avg: {self.best_avg:.1f}",
            f"High Score:   {self.high_score}",
            f"Current Pts:  {env_obj.score}",
            f"Epsilon:      {epsilon:.3f}",
            f"Buffer Size:  {len(self.replay_buffer):,}",
        ]
        for i, line in enumerate(stats):
            txt = self.font.render(line, True, (210, 220, 235))
            self.screen.blit(txt, (ox, oy + HEIGHT * cell + 18 + i * 21))

        # Draw Afterstate Value Bars for candidate moves
        if self.last_values is not None:
            bx, by = 320, 80
            lbl = self.font.render("Candidate Afterstate Values V(s') - Highest Wins:", True, (180, 185, 200))
            self.screen.blit(lbl, (bx, by - 25))

            vals = self.last_values
            v_min, v_max = vals.min(), vals.max()
            span = max(v_max - v_min, 1e-4)

            for i, val in enumerate(vals[:22]):
                norm = (val - v_min) / span
                bar_w = int(20 + norm * 520)
                is_best = (i == self.last_chosen)
                bar_col = (80, 250, 140) if is_best else (65, 130, 230)
                y_pos = by + i * 24
                pygame.draw.rect(self.screen, bar_col, (bx, y_pos, bar_w, 18), border_radius=4)
                tag = self.font.render(f"Move {i:02d}: {val:6.2f}", True, (235, 240, 245))
                self.screen.blit(tag, (bx + bar_w + 10, y_pos + 1))

        pygame.display.flip()

    def train(self):
        for ep in range(1, self.total_episodes + 1):
            if not self.running:
                break

            env = PuyoEnv(animate=False)
            prev_b, prev_m = encode_afterstate(env)
            epsilon = max(0.02, 0.40 * (0.9996 ** ep))
            done = False

            while not done:
                candidates = get_valid_afterstates(env)
                _, next_env, reward, done, score, _, next_b, next_m = self._select_action(candidates, epsilon)

                self.replay_buffer.append((prev_b, prev_m, reward, next_b, next_m, float(done)))
                env = next_env
                prev_b, prev_m = next_b, next_m

                self._train_step()

            self.scores_history.append(env.score)
            self.high_score = max(self.high_score, env.score)
            avg_100 = float(np.mean(self.scores_history))

            # Save best model automatically when 100-game rolling average improves
            if len(self.scores_history) >= 50 and avg_100 > self.best_avg:
                self.best_avg = avg_100
                th.save(self.online_net.state_dict(), "puyo_afterstate_best.pt")

            if ep % 15 == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if ep % self.render_every == 0:
                self._draw_viz(env, ep, epsilon, finished=False)

            if ep % 100 == 0:
                print(
                    f"Ep {ep:5d} | 100-Avg: {avg_100:7.1f} | Best Avg: {self.best_avg:7.1f} | High: {self.high_score} | Eps: {epsilon:.3f}"
                )

        th.save(self.online_net.state_dict(), "puyo_afterstate_final.pt")

        # Keep Pygame window open at the end
        self._draw_viz(env, ep, epsilon, finished=True)
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    trainer = GPUAfterstateTrainer(total_episodes=15_000, render_every=15)
    trainer.train()
