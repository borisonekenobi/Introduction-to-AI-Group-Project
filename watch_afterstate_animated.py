import operator

import numpy as np
import pygame
import torch as th

from puyo_env import HEIGHT, PuyoEnv, WIDTH
from train_afterstate_gpu import AfterstateValueNet, get_valid_afterstates, PLACEMENTS


class AnimatedAfterstateVisualizer:
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
        pygame.display.set_caption(f"Puyo Puyo Animated GPU Agent ({weights_path})")
        self.font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 18, bold=True)

        self.paused = False
        self.anim_fps = 60  # Animation frames per second (use UP/DOWN to adjust)
        self.games_played = 0
        self.high_score = 0
        self.completed_scores = []
        self.longest_chain_game = 0
        self.longest_chain_all_time = 0

        self.last_candidates = []
        self.last_q_values = []
        self.last_chosen_idx = 0

        # Animation state machine: "THINK" -> "MOVE" -> "FLASH" -> "GRAVITY"
        self.phase = "THINK"
        self.move_queue = []
        self.flashing_cells = set()
        self.flash_ticks = 0
        self.pending_chain = 0

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

    def _choose_best_action_idx(self, beam_width=4):
        """3-ply GPU beam search returning the chosen placement action_idx."""
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
        best_action_idx, _, _, _, _, _, _, _ = best_cand

        for q1, idx1, cand1 in ranked1[:beam_width]:
            act1, sim1, r1, d1, _, _, _, _ = cand1
            if d1:
                continue

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
                best_action_idx = act1

        self.last_chosen_idx = best_idx
        return best_action_idx

    def _build_animation_steps(self, action_idx):
        """Convert a target placement into individual rotation, slide, and fall steps."""
        target_x, target_rot = PLACEMENTS[action_idx]
        steps = []

        if self.env.current is None:
            return steps

        # Calculate shortest rotation path
        curr_rot = self.env.current["rotation"]
        rot_diff = (target_rot - curr_rot) % 4
        if rot_diff == 3:
            steps.append("rotate_ccw")
        else:
            for _ in range(rot_diff):
                steps.append("rotate_cw")

        # Horizontal slide steps
        curr_x = self.env.current["x"]
        dx = 1 if target_x > curr_x else -1
        for _ in range(abs(target_x - curr_x)):
            steps.append("right" if dx > 0 else "left")

        # Smooth fast-fall steps down to the stack
        for _ in range(HEIGHT):
            steps.append("fall")

        return steps

    def _lock_piece_without_resolving(self):
        """Place current falling pair onto board and apply initial gravity without popping yet."""
        p = self.env.current
        if p is None:
            return
        cells = self.env._cells(p["x"], p["y"], p["rotation"])
        for (x, y), color in zip(cells, p["colors"]):
            if 0 <= y < HEIGHT:
                self.env.board[y][x] = color
        self.env.current = None
        self.env._apply_gravity()
        self.pending_chain = 0

    def _advance_animation(self):
        """Advance one micro-step of movement, chain flashing, or gravity cascade."""
        if self.env.game_over:
            self.games_played += 1
            self.completed_scores.append(self.env.score)
            self.longest_chain_game = 0
            self.env.reset()
            self.phase = "THINK"
            return

        if self.phase == "THINK":
            if self.env.current is None:
                self.env.spawn()
                if self.env.game_over:
                    return
            chosen_act = self._choose_best_action_idx()
            self.move_queue = self._build_animation_steps(chosen_act)
            self.phase = "MOVE"

        elif self.phase == "MOVE":
            if not self.move_queue:
                self._lock_piece_without_resolving()
                self._check_for_matches()
                return

            cmd = self.move_queue.pop(0)
            if cmd == "rotate_cw":
                self.env._rotate(1)
            elif cmd == "rotate_ccw":
                self.env._rotate(-1)
            elif cmd == "left":
                self.env._move(-1, 0)
            elif cmd == "right":
                self.env._move(1, 0)
            elif cmd == "fall":
                # Move down 2 rows per frame for crisp, fluid falling
                moved1 = self.env._move(0, 1)
                moved2 = self.env._move(0, 1) if moved1 else False
                if not moved1 or not moved2:
                    self.move_queue.clear()
                    self._lock_piece_without_resolving()
                    self._check_for_matches()

        elif self.phase == "FLASH":
            self.flash_ticks -= 1
            if self.flash_ticks <= 0:
                # Pop the flashing groups and award chain points
                popped = len(self.flashing_cells)
                for x, y in self.flashing_cells:
                    self.env.board[y][x] = None
                self.pending_chain += 1
                self.env.chain = self.pending_chain
                self.longest_chain_game = max(self.longest_chain_game, self.pending_chain)
                self.longest_chain_all_time = max(self.longest_chain_all_time, self.pending_chain)
                chain_bonus = 2 ** min(self.pending_chain - 1, 8)
                self.env.score += popped * 10 * chain_bonus
                self.high_score = max(self.high_score, self.env.score)

                self.flashing_cells.clear()
                self.phase = "GRAVITY"

        elif self.phase == "GRAVITY":
            # Let unsupported puyos fall after a clear, then check for cascade chains
            self.env._apply_gravity()
            self._check_for_matches()

    def _check_for_matches(self):
        """Find any 4+ connected color groups; flash them if found, else spawn next pair."""
        groups = [g for g in self.env._components() if len(g) >= 4]
        if groups:
            self.flashing_cells = {cell for grp in groups for cell in grp}
            # Hold flash for ~4 frames so the player clearly sees which puyos are popping
            self.flash_ticks = max(3, self.anim_fps // 4)
            self.phase = "FLASH"
        else:
            self.env.spawn()
            self.phase = "THINK"

    def _draw_puyo(self, cx, cy, radius, color_code, flashing=False):
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
            if flashing:
                # Bright pulsing outline + inner glow when a group is about to pop
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius + 2, 3)

    def _draw_board_and_stats(self):
        cell = 24
        ox, oy = 30, 85

        pygame.draw.rect(
            self.screen,
            (70, 75, 92),
            (ox - 4, oy - 4, WIDTH * cell + 8, HEIGHT * cell + 8),
            2,
        )

        # Draw settled board puyos
        for y in range(HEIGHT):
            for x in range(WIDTH):
                rect = (ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2)
                pygame.draw.rect(self.screen, (28, 31, 40), rect, border_radius=4)
                piece = self.env.board[y][x]
                if piece is not None:
                    is_flashing = (x, y) in self.flashing_cells
                    self._draw_puyo(
                        ox + x * cell + cell // 2,
                        oy + y * cell + cell // 2,
                        cell // 2 - 2,
                        piece,
                        flashing=is_flashing,
                        )

        # Draw active falling pair in real time as it rotates, slides, and drops
        if self.env.current is not None:
            p = self.env.current
            cells = self.env._cells(p["x"], p["y"], p["rotation"])
            for (cx, cy), color in zip(cells, p["colors"]):
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    self._draw_puyo(
                        ox + cx * cell + cell // 2,
                        oy + cy * cell + cell // 2,
                        cell // 2 - 2,
                        color,
                        )

        # Draw NEXT queue on side
        qx = ox + WIDTH * cell + 22
        lbl_next = self.font.render("NEXT", True, (180, 185, 200))
        self.screen.blit(lbl_next, (qx, oy))
        for idx, pair in enumerate(list(self.env.pair_queue)[:2]):
            p0, p1 = pair
            base_y = oy + 32 + idx * 65
            self._draw_puyo(qx + 18, base_y, 9, p1)
            self._draw_puyo(qx + 18, base_y + 22, 9, p0)

        # Chain banner when popping
        if self.env.chain > 0:
            chain_txt = self.title_font.render(f"{self.env.chain} CHAIN!", True, (255, 230, 90))
            self.screen.blit(chain_txt, (qx, oy + 175))

        avg_score = np.mean(self.completed_scores) if self.completed_scores else 0.0
        low_score = np.min(self.completed_scores) if self.completed_scores else 0

        stats = [
            f"Score:                    {self.env.score}",
            f"Avg Score:                {avg_score:.1f}",
            f"High Score:               {self.high_score}",
            f"Low Score:                {low_score}",
            f"Last Chain:               {self.env.chain}",
            f"Longest Chain (Game):     {self.longest_chain_game}",
            f"Longest Chain (All Time): {self.longest_chain_all_time}",
            f"Games:                    {self.games_played}",
            f"Anim Speed:               {self.anim_fps} FPS",
            "",
            "[SPACE] Pause/Resume",
            "[RIGHT] Step 1 frame",
            "[UP/DN] Change speed",
            "[R]     Reset board",
        ]
        for i, line in enumerate(stats):
            color = (140, 220, 170) if i < 8 else (150, 155, 170)
            txt = self.font.render(line, True, color)
            self.screen.blit(txt, (ox, oy + HEIGHT * cell + 12 + i * 17))

    def _draw_viz(self):
        self.screen.fill((18, 20, 26))

        state_tag = "PAUSED" if self.paused else f"{self.phase} ({self.anim_fps} FPS)"
        header_color = (255, 200, 90) if self.paused else (100, 240, 150)
        header = self.title_font.render(
            f"Animated GPU Afterstate Agent [{state_tag}]", True, header_color
        )
        self.screen.blit(header, (30, 25))

        self._draw_board_and_stats()

        if len(self.last_q_values) > 0:
            bx, by = 330, 85
            lbl = self.font.render(
                "3-Ply Evaluated Afterstate Moves - Highest Chosen:",
                True,
                (180, 185, 200),
            )
            self.screen.blit(lbl, (bx, by - 25))

            vals = self.last_q_values
            v_min, v_max = vals.min(), vals.max()
            span = max(v_max - v_min, 1e-4)

            for i, (cand, val) in enumerate(zip(self.last_candidates, vals)):
                action_idx, _, _, _, _, _, _, _ = cand
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
                        self.anim_fps = min(60, self.anim_fps + 3)
                    elif event.key == pygame.K_DOWN:
                        self.anim_fps = max(3, self.anim_fps - 3)
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self.longest_chain_game = 0
                        self.phase = "THINK"
                        self.move_queue.clear()
                        self.flashing_cells.clear()

            if not self.paused or step_once:
                self._advance_animation()

            self._draw_viz()
            clock.tick(self.anim_fps if not self.paused else 30)

        pygame.quit()


if __name__ == "__main__":
    viewer = AnimatedAfterstateVisualizer("puyo_afterstate_best.pt")
    viewer.run()
