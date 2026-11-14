import operator

import numpy as np
import pygame
import torch as th

from puyo_env import HEIGHT, PuyoEnv, WIDTH
from train_afterstate_gpu import AfterstateValueNet, get_valid_afterstates, PLACEMENTS


class AfterstatePlayerVisualizer:
    def __init__(self, weights_path="puyo_afterstate_best.pt", width=1120, height=700):
        self.width = width
        self.height = height
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.net = AfterstateValueNet().to(self.device)
        self.net.load_state_dict(th.load(weights_path, map_location=self.device))
        self.net.eval()

        self.env = PuyoEnv(animate=False)
        self.gamma = 0.985

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Puyo Puyo GPU Afterstate Agent ({weights_path})")
        self.font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 18, bold=True)

        self.paused = False
        self.fps = 30
        self.games_played = 0
        self.high_score = 0
        self.completed_scores = []
        self.last_candidates = []
        self.last_q_values = []
        self.last_chosen_idx = 0

    def _score_candidates_gpu(self, candidates):
        """Batch-evaluate candidate afterstates on GPU + spawn-chute survival guard."""
        boards_list = []
        metas_list = []
        rewards_list = []
        dones_list = []
        danger_penalties = []

        for action_idx, sim_env, step_reward, done, score, chain, b_arr, m_arr in candidates:
            boards_list.append(b_arr)
            metas_list.append(m_arr)
            rewards_list.append(step_reward)
            dones_list.append(1.0 if done else 0.0)

            # Check heights of critical centre columns (x = 1, 2, 3)
            penalty = 0.0
            for cx, weight in ((1, 0.4), (2, 1.2), (3, 0.4)):
                col_h = 0
                for y in range(HEIGHT):
                    if sim_env.board[y][cx] is not None:
                        col_h = HEIGHT - y
                        break
                if col_h >= 7:
                    penalty += weight * ((col_h - 6) ** 1.8)
            danger_penalties.append(penalty)

        boards = th.as_tensor(np.stack(boards_list), device=self.device)
        metas = th.as_tensor(np.stack(metas_list), device=self.device)
        rewards = np.array(rewards_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=np.float32)
        penalties = np.array(danger_penalties, dtype=np.float32)

        with th.no_grad():
            future_vals = self.net(boards, metas).cpu().numpy()

        return rewards + (1.0 - dones) * self.gamma * future_vals - penalties

    def _choose_best_move(self, beam_width=4):
        """3-ply GPU beam search using all 3 visible pairs (CURRENT + NEXT 1 + NEXT 2)."""
        ply1_candidates = get_valid_afterstates(self.env)
        ply1_q = self._score_candidates_gpu(ply1_candidates)

        self.last_candidates = ply1_candidates
        self.last_q_values = ply1_q

        ranked1 = sorted(
            zip(ply1_q, range(len(ply1_candidates)), ply1_candidates),
            key=operator.itemgetter(0),
            reverse=True,
        )

        best_total_q = -1e9
        _, best_idx, best_cand = next(iter(ranked1))
        best_sim_env = best_cand[1]

        for q1, idx1, cand1 in ranked1[:beam_width]:
            _, sim1, r1, d1, _, _, _, _ = cand1
            if d1:
                continue

            # Ply 2: NEXT pair 1
            ply2_candidates = get_valid_afterstates(sim1)
            ply2_q = self._score_candidates_gpu(ply2_candidates)
            ranked2 = sorted(
                zip(ply2_q, ply2_candidates),
                key=operator.itemgetter(0),
                reverse=True,
            )

            best_ply2_val = -1e9
            for q2, cand2 in ranked2[:beam_width]:
                _, sim2, r2, d2, _, _, _, _ = cand2
                if d2:
                    continue

                # Ply 3: NEXT pair 2
                ply3_candidates = get_valid_afterstates(sim2)
                ply3_q = self._score_candidates_gpu(ply3_candidates)
                best_q3 = float(np.max(ply3_q))

                val2 = r2 + self.gamma * best_q3
                if val2 > best_ply2_val:
                    best_ply2_val = val2

            total_q = r1 + self.gamma * best_ply2_val
            if total_q > best_total_q:
                best_total_q = total_q
                best_idx = idx1
                best_sim_env = sim1

        self.last_chosen_idx = best_idx
        return best_sim_env



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

    def _draw_board_and_stats(self):
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
                piece = self.env.board[y][x]
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
        if self.env.current:
            c0, c1 = self.env.current["colors"]
            self._draw_puyo(qx + 18, oy + 32, 10, c1)
            self._draw_puyo(qx + 18, oy + 56, 10, c0)

        lbl_next = self.font.render("NEXT", True, (180, 185, 200))
        self.screen.blit(lbl_next, (qx, oy + 90))
        for idx, pair in enumerate(list(self.env.pair_queue)[:2]):
            p0, p1 = pair
            base_y = oy + 122 + idx * 65
            self._draw_puyo(qx + 18, base_y, 9, p1)
            self._draw_puyo(qx + 18, base_y + 22, 9, p0)

        avg_score = np.mean(self.completed_scores) if self.completed_scores else 0.0
        low_score = np.min(self.completed_scores) if self.completed_scores else 0

        stats = [
            f"Score:      {self.env.score}",
            f"Avg Score:  {avg_score:.1f}",
            f"High Score: {self.high_score}",
            f"Low Score:  {low_score}",
            f"Last Chain: {self.env.chain}",
            f"Games:      {self.games_played}",
            f"Speed:      {self.fps} moves/sec",
            "",
            "[SPACE] Pause/Resume",
            "[RIGHT] Step 1 move",
            "[UP/DN] Change speed",
            "[R]     Reset board",
        ]
        for i, line in enumerate(stats):
            color = (140, 220, 170) if i < 6 else (150, 155, 170)
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (ox, oy + HEIGHT * cell + 16 + i * 19))

    def _draw_viz(self):
        self.screen.fill((18, 20, 26))

        state_tag = "PAUSED" if self.paused else f"RUNNING ({self.fps} FPS)"
        header_color = (255, 200, 90) if self.paused else (100, 240, 150)
        header = self.title_font.render(
            f"Afterstate Value Net Playback ({self.device}) [{state_tag}]",
            True,
            header_color,
        )
        self.screen.blit(header, (30, 25))

        self._draw_board_and_stats()

        if len(self.last_q_values) > 0:
            bx, by = 330, 85
            lbl = self.font.render(
                "Evaluated Afterstate Moves (r + gamma * V(s')) - Highest Chosen:",
                True,
                (180, 185, 200),
            )
            self.screen.blit(lbl, (bx, by - 25))

            vals = self.last_q_values
            v_min, v_max = vals.min(), vals.max()
            span = max(v_max - v_min, 1e-4)

            for i, (cand, val) in enumerate(zip(self.last_candidates, vals)):
                action_idx = cand[0]
                tx, trot = PLACEMENTS[action_idx]
                norm = (val - v_min) / span
                bar_w = int(20 + norm * 480)
                is_best = i == self.last_chosen_idx
                bar_col = (80, 250, 140) if is_best else (65, 130, 230)
                y_pos = by + i * 25
                pygame.draw.rect(
                    self.screen, bar_col, (bx, y_pos, bar_w, 19), border_radius=4
                )
                tag = self.font.render(
                    f"x={tx} r={trot} | Q={val:6.2f}", True, (235, 240, 245)
                )
                self.screen.blit(tag, (bx + bar_w + 10, y_pos + 2))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True

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
                        self.fps = min(30, self.fps + 1)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 1)
                    elif event.key == pygame.K_r:
                        self.env.reset()

            if not self.paused or step_once:
                self.env = self._choose_best_move()
                self.high_score = max(self.high_score, self.env.score)
                if self.env.game_over:
                    self.games_played += 1
                    self.completed_scores.append(self.env.score)
                    self.env.reset()

            self._draw_viz()
            clock.tick(self.fps if not self.paused else 30)

        pygame.quit()


if __name__ == "__main__":
    viewer = AfterstatePlayerVisualizer("puyo_afterstate_best.pt")
    viewer.run()
