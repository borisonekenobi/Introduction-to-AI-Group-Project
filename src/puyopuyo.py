"""
Puyo Puyo (simplified) for human play and AI experimentation.

Install:
    pip install pygame

Run:
    python puyopuyo.py

Controls:
    Left/Right : move pair
    Down       : soft drop
    Up / X     : rotate clockwise
    Z          : rotate counter-clockwise
    Space      : hard drop
    R          : restart
    Esc        : quit

AI interface:
    env = PuyoEnv(seed=123)
    observation = env.observe()
    reward, done, info = env.step(action)

Actions: "left", "right", "down", "rotate_cw", "rotate_ccw", "drop", "wait"

The observation contains the board, active pair, and the next two pairs. The
game logic is independent of pygame, so it can also be used headlessly.
This is a compact Puyo-style implementation, not a frame-perfect clone:
garbage puyos, fever mode, and versus rules are intentionally omitted.
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Optional

# Board coordinates: x=0..5, y=0..11; y=0 is the top.
WIDTH, HEIGHT = 6, 12
COLORS = ("R", "G", "B", "Y")
EMPTY = None

# Child-puyo offsets relative to the pivot for each orientation.
# 0=above, 1=right, 2=below, 3=left.
OFFSETS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTIONS = ("left", "right", "down", "rotate_cw", "rotate_ccw", "drop", "wait")


class PuyoEnv:
    """Game state and rules. No pygame dependency."""

    def __init__(self, seed=None, animate=False):
        self.rng = random.Random(seed)
        self.animate = animate
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.board = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.score = 0
        self.chain = 0
        self.game_over = False
        self.pair_queue = deque(self._new_pair() for _ in range(3))
        self.current = None
        self.resolving = False
        self.clearing_group = []
        self.clear_timer = 0
        self.clear_delay = 350  # milliseconds
        self.spawn()
        return self.observe()

    def _new_pair(self):
        return (self.rng.choice(COLORS), self.rng.choice(COLORS))

    def _ensure_queue(self):
        while len(self.pair_queue) < 3:
            self.pair_queue.append(self._new_pair())

    def spawn(self):
        self._ensure_queue()
        self.current = {
            "x": 2, "y": 1, "rotation": 0,
            "colors": self.pair_queue.popleft()
        }
        self._ensure_queue()
        if not self._valid(self.current["x"], self.current["y"],
                           self.current["rotation"]):
            self.game_over = True

    def _cells(self, x, y, rotation):
        dx, dy = OFFSETS[rotation % 4]
        return ((x, y), (x + dx, y + dy))

    def _valid(self, x, y, rotation):
        for cx, cy in self._cells(x, y, rotation):
            if cx < 0 or cx >= WIDTH or cy < 0 or cy >= HEIGHT:
                return False
            if self.board[cy][cx] is not None:
                return False
        return True

    def _move(self, dx, dy):
        if self.current is None or self.game_over:
            return False
        p = self.current
        if self._valid(p["x"] + dx, p["y"] + dy, p["rotation"]):
            p["x"] += dx
            p["y"] += dy
            return True
        return False

    def _rotate(self, direction):
        if self.current is None or self.game_over:
            return False
        p = self.current
        new_rot = (p["rotation"] + direction) % 4
        # Small wall/floor kicks make rotation near edges less frustrating.
        for kx, ky in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if self._valid(p["x"] + kx, p["y"] + ky, new_rot):
                p["x"] += kx
                p["y"] += ky
                p["rotation"] = new_rot
                return True
        return False

    def _land(self):
        p = self.current

        if p is None:
            return

        cells = self._cells(
            p["x"],
            p["y"],
            p["rotation"]
        )

        for (x, y), color in zip(cells, p["colors"]):
            if 0 <= y < HEIGHT:
                self.board[y][x] = color

        self.current = None

        # Always apply gravity immediately.
        self._apply_gravity()

        if self.animate:
            self.chain = 0
            self._start_animated_resolution()
        else:
            gained, chains = self._resolve()
            self.score += gained
            self.chain = chains
            self.spawn()

    def _apply_gravity(self):
        """Make every column fall independently toward the bottom."""
        moved = False

        for x in range(WIDTH):
            # Collect all occupied cells, preserving their order.
            puyos = [
                self.board[y][x]
                for y in range(HEIGHT)
                if self.board[y][x] is not None
            ]

            # Place them at the bottom of the column.
            new_column = (
                [None] * (HEIGHT - len(puyos)) + puyos
            )

            for y in range(HEIGHT):
                if self.board[y][x] != new_column[y]:
                    moved = True

                self.board[y][x] = new_column[y]

        return moved

    def _components(self):
        seen = set()
        groups = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                color = self.board[y][x]
                if color is None or (x, y) in seen:
                    continue
                stack = [(x, y)]
                seen.add((x, y))
                group = []
                while stack:
                    cx, cy = stack.pop()
                    group.append((cx, cy))
                    for nx, ny in ((cx+1, cy), (cx-1, cy),
                                   (cx, cy+1), (cx, cy-1)):
                        if (0 <= nx < WIDTH and 0 <= ny < HEIGHT
                                and (nx, ny) not in seen
                                and self.board[ny][nx] == color):
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                groups.append(group)
        return groups

    def _resolve(self):
        total_score = 0
        chain_count = 0

        while True:
            groups = [
                group
                for group in self._components()
                if len(group) >= 4
            ]

            if not groups:
                break

            chain_count += 1

            popped = sum(len(group) for group in groups)

            # Remove matching groups.
            for group in groups:
                for x, y in group:
                    self.board[y][x] = None

            # Award points.
            chain_bonus = 2 ** min(chain_count - 1, 8)
            total_score += popped * 10 * chain_bonus

            # Let unsupported Puyos fall independently.
            self._apply_gravity()

        return total_score, chain_count

    def step(self, action):
        """Apply one discrete action; return (reward, done, info)."""
        if self.resolving:
            return 0, False, {
                "score": self.score,
                "chain": self.chain,
                "resolving": True
            }
        if self.game_over:
            return 0, True, {"score": self.score, "chain": self.chain}
        reward = 0
        if action == "left":
            self._move(-1, 0)
        elif action == "right":
            self._move(1, 0)
        elif action == "down":
            if not self._move(0, 1):
                before = self.score
                self._land()
                reward = self.score - before
        elif action == "rotate_cw":
            self._rotate(1)
        elif action == "rotate_ccw":
            self._rotate(-1)
        elif action == "drop":
            while self._move(0, 1):
                pass
            before = self.score
            self._land()
            reward = self.score - before
        elif action == "wait":
            pass
        else:
            raise ValueError(f"Unknown action: {action}")
        return reward, self.game_over, {
            "score": self.score, "chain": self.chain,
            "game_over": self.game_over
        }

    def observe(self):
        """Return a fresh, serializable observation for an AI agent."""
        self._ensure_queue()
        return {
            "board": tuple(tuple(row) for row in self.board),
            "current": None if self.current is None else {
                "x": self.current["x"], "y": self.current["y"],
                "rotation": self.current["rotation"],
                "colors": tuple(self.current["colors"]),
            },
            "next": tuple(tuple(pair) for pair in list(self.pair_queue)[:2]),
            "score": self.score,
            "chain": self.chain,
            "game_over": self.game_over,
            "width": WIDTH,
            "height": HEIGHT,
        }

    def legal_actions(self):
        """Actions that can be applied now (drop/wait always included)."""
        if self.game_over or self.current is None:
            return []
        p = self.current
        legal = ["drop", "wait"]
        if self._valid(p["x"] - 1, p["y"], p["rotation"]):
            legal.append("left")
        if self._valid(p["x"] + 1, p["y"], p["rotation"]):
            legal.append("right")
        if self._valid(p["x"], p["y"] + 1, p["rotation"]):
            legal.append("down")
        if any(self._valid(p["x"] + kx, p["y"] + ky,
                           (p["rotation"] + d) % 4)
               for d in (-1, 1) for kx, ky in
               ((0, 0), (-1, 0), (1, 0), (0, -1))):
            legal.extend(("rotate_cw", "rotate_ccw"))
        return legal

    def _start_animated_resolution(self):
        """Begin resolving groups without blocking the game loop."""
        self.resolving = True
        self.clearing_group = []
        self.clear_timer = 0
        self._find_next_group()


    def _find_next_group(self):
        """Find the next group that should disappear."""
        groups = [
            group for group in self._components()
            if len(group) >= 4
        ]

        if groups:
            # Highlight one group at a time.
            self.clearing_group = groups[0]
            self.clear_timer = self.clear_delay
        else:
            # No more matches: finish resolving and spawn.
            self.resolving = False
            self.clearing_group = []
            self.spawn()


    def update(self, dt):
        """Advance animated resolution using elapsed milliseconds."""
        if not self.resolving:
            return

        self.clear_timer -= dt

        if self.clear_timer <= 0:
            # Remove the highlighted group.
            for x, y in self.clearing_group:
                self.board[y][x] = None

            # Award points.
            popped = len(self.clearing_group)
            self.chain += 1

            chain_bonus = 2 ** min(self.chain - 1, 8)
            self.score += popped * 10 * chain_bonus

            # Apply gravity independently to every column.
            self._apply_gravity()

            # Find the next group, if one exists.
            self._find_next_group()


def choose_random_action(env):
    """Tiny example agent; replace with your own policy."""
    return random.choice(env.legal_actions())


def choose_greedy_drop(env):
    """Example placement agent: test reachable pair placements and score them.

    This evaluates the immediate result only. It is a useful baseline, not a
    strong Puyo strategy. It uses the same public observation as other agents.
    """
    best_action = "drop"
    best_value = float("-inf")
    # Search horizontal positions and four orientations by simulating controls.
    # BFS over movement/rotation states, then evaluate each reachable drop.
    start = env.current
    if start is None:
        return "wait"
    queue = deque([(start["x"], start["y"], start["rotation"], [])])
    visited = {(start["x"], start["y"], start["rotation"])}
    candidates = []
    while queue:
        x, y, rot, path = queue.popleft()
        # A placement is available if it cannot fall further.
        if not env._valid(x, y + 1, rot):
            candidates.append((x, y, rot, path + ["drop"]))
        for action, nx, ny, nr in (
            ("left", x-1, y, rot), ("right", x+1, y, rot),
            ("down", x, y+1, rot),
            ("rotate_cw", x, y, (rot+1) % 4),
            ("rotate_ccw", x, y, (rot-1) % 4),
        ):
            state = (nx, ny, nr)
            if state not in visited and env._valid(nx, ny, nr):
                visited.add(state)
                queue.append((nx, ny, nr, path + [action]))

    original_board = [row[:] for row in env.board]
    original_score = env.score
    original_chain = env.chain
    original_current = dict(env.current)
    for x, y, rot, path in candidates:
        sim = PuyoEnv(seed=0)
        sim.board = [row[:] for row in original_board]
        sim.current = {"x": x, "y": y, "rotation": rot,
                       "colors": original_current["colors"]}
        sim.pair_queue = deque(env.pair_queue)
        sim.score = original_score
        sim.game_over = False
        sim._land()
        heights = [sum(1 for row in sim.board if row[c] is not None)
                   for c in range(WIDTH)]
        # Prefer clears, then low/flat boards; penalize near-top columns.
        value = (sim.score - original_score) * 10
        value -= sum(h * h for h in heights) * 0.7
        value -= max(heights) * 3
        if sim.game_over:
            value -= 10000
        if value > best_value:
            best_value, best_action = value, (path[0] if path else "drop")
    return best_action


def run_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install pygame first: pip install pygame") from exc

    pygame.init()
    pygame.display.set_caption("Puyo Puyo — Human / AI")
    cell = 38
    margin = 22
    board_w, board_h = WIDTH * cell, HEIGHT * cell
    screen = pygame.display.set_mode((board_w + 250, board_h + margin * 2))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    small = pygame.font.SysFont("consolas", 16)
    env = PuyoEnv(animate=True)
    ai_mode = False
    fall_timer = 0
    fall_interval = 550
    ai_timer = 0
    ai_interval = 120
    colors = {
        "R": (230, 75, 75), "G": (70, 190, 100),
        "B": (75, 130, 235), "Y": (245, 210, 65),
    }
    bg = (22, 25, 34)
    board_bg = (38, 43, 56)

    def draw_puyo(cx, cy, color, radius=16):
        px = margin + cx * cell + cell // 2
        py = margin + cy * cell + cell // 2
        pygame.draw.circle(screen, colors[color], (px, py), radius)
        pygame.draw.circle(screen, (255, 255, 255), (px-5, py-6), 4)

    running = True
    while running:
        dt = clock.tick(60)
        env.update(dt)
        fall_timer += dt
        ai_timer += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_TAB:
                    ai_mode = not ai_mode
                elif not ai_mode and not env.game_over:
                    key_actions = {
                        pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
                        pygame.K_DOWN: "down", pygame.K_UP: "rotate_cw",
                        pygame.K_x: "rotate_cw", pygame.K_z: "rotate_ccw",
                        pygame.K_SPACE: "drop",
                    }
                    if event.key in key_actions:
                        env.step(key_actions[event.key])

        if not env.game_over:
            if ai_mode and ai_timer >= ai_interval:
                ai_timer = 0
                # Swap this function for your own agent.
                env.step(choose_greedy_drop(env))
            elif not ai_mode and fall_timer >= fall_interval:
                fall_timer = 0
                env.step("down")
        else:
            fall_timer = 0

        screen.fill(bg)
        pygame.draw.rect(screen, board_bg,
                         (margin, margin, board_w, board_h), border_radius=8)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                rect = (margin + x*cell + 1, margin + y*cell + 1,
                        cell-2, cell-2)
                pygame.draw.rect(screen, (48, 54, 68), rect, border_radius=4)
                value = env.board[y][x]
                if value:
                    draw_puyo(x, y, value)

                    if (x, y) in env.clearing_group:
                        px = margin + x * cell + cell // 2
                        py = margin + y * cell + cell // 2

                        # Pulsating white outline.
                        pulse = 19 + int(3 * abs(
                            pygame.math.Vector2(
                                pygame.time.get_ticks() % 700, 0
                            ).x / 700 - 0.5
                        ))

                        pygame.draw.circle(
                            screen,
                            (255, 255, 255),
                            (px, py),
                            pulse,
                            3
                        )

        if env.current and not env.game_over:
            for (x, y), color in zip(
                    env._cells(env.current["x"], env.current["y"],
                               env.current["rotation"]),
                    env.current["colors"]):
                draw_puyo(x, y, color)

        panel_x = margin + board_w + 24
        screen.blit(font.render("PUYO PUYO", True, (245, 245, 250)),
                    (panel_x, 30))
        screen.blit(font.render(f"Score: {env.score}", True, (245, 245, 250)),
                    (panel_x, 80))
        screen.blit(font.render("Next", True, (245, 245, 250)),
                    (panel_x, 140))
        for i, pair in enumerate(list(env.pair_queue)[:2]):
            draw_x = panel_x - margin
            for j, color in enumerate(pair):
                px = draw_x + 32 + j*38
                py = 205 + i*75
                pygame.draw.circle(screen, colors[color], (px, py), 16)
                pygame.draw.circle(screen, (255,255,255), (px-5,py-6), 4)
        mode = "AI" if ai_mode else "Human"
        screen.blit(small.render(f"Mode: {mode} (TAB)", True,
                                 (190, 200, 215)), (panel_x, 330))
        controls = ["Arrows: move/drop", "Up/X: rotate CW",
                    "Z: rotate CCW", "Space: hard drop",
                    "R: restart", "Esc: quit"]
        for i, line in enumerate(controls):
            screen.blit(small.render(line, True, (190, 200, 215)),
                        (panel_x, 375 + i*25))
        if env.game_over:
            overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 175))
            screen.blit(overlay, (margin, margin))
            msg = font.render("GAME OVER", True, (255, 230, 230))
            screen.blit(msg, (margin + 55, margin + board_h//2 - 15))
            sub = small.render("Press R to restart", True, (255,255,255))
            screen.blit(sub, (margin + 78, margin + board_h//2 + 20))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run a short headless random-agent demo.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.headless:
        env = PuyoEnv(seed=args.seed)
        while not env.game_over:
            env.step(choose_random_action(env))
        print(f"Game over. Score: {env.score}")
    else:
        run_pygame()
