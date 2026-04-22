from typing import Tuple

import numpy as np


class UAV3D:
    """
    3D UAV motion planning benchmark from JAIR 2022 paper.

    "Robust Control for Dynamical Systems with Non-Gaussian Noise via Formal Abstractions"
    by Thom Badings et al.

    Reference implementation: https://github.com/LAVA-LAB/DynAbs

    State: [x_pos, x_vel, y_pos, y_vel, z_pos, z_vel] in R^6
    Action: [fx, fy, fz] in [-4, 4]^3 (force/acceleration commands)

    Dynamics:
        x_{k+1} = A x_k + B u_k + w_k
    where A/B are the double-integrator matrices per axis.

    Goal: {11 <= x <= 15, 1 <= y <= 5, -7 <= z <= -3}
    Obstacles: 14 3D rectangular obstacles (critical regions).
    """

    def __init__(
        self,
        gaussian_noise: bool = True,
        noise_std: float = 0.15,
        max_steps: int = 64,
    ):
        self.max_steps = max_steps
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std

        self.A = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.B = np.array([
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.u_min = -4.0
        self.u_max = 4.0

        self.goal_min = np.array([11.0, 1.0, -7.0])
        self.goal_max = np.array([15.0, 5.0, -3.0])
        self.goal_center = 0.5 * (self.goal_min + self.goal_max)

        self.obstacles = [
            (-11, -5, -1, 9, -7, -5),
            (-11, -5, 5, 9, -5, 5),
            (-11, -5, -1, 3, -5, 3),
            (-1, 3, 1, 9, -7, -1),
            (-1, 3, 1, 9, 3, 5),
            (-1, 3, 1, 3, -1, 3),
            (-1, 3, 7, 9, -1, 3),
            (-1, 3, -3, 1, -7, 7),
            (3, 9, -3, 1, -7, -1),
            (-11, -7, -5, -1, -7, 1),
            (-1, 3, -9, -3, -7, -5),
            (-1, 3, -9, -3, 3, 7),
            (11, 15, -9, -5, -7, -5),
            (9, 15, 5, 9, -7, 1),
        ]

        self._state = np.zeros(6, dtype=np.float32)
        self._done = False
        self._step_count = 0

        # Canonical start position (left edge, moderate height): far from goal.
        self.start_state = np.array([-14.0, 0.0, 6.0, 0.0, -6.0, 0.0], dtype=np.float32)

        # Easy-curriculum start in a clear slab with >= 1 unit obstacle buffer
        # in every direction. Goal center (13, 3, -5) is reachable by a pure
        # +x push; the goal region extends to z=-3 so no z motion is required.
        self.easy_start_state = np.array([7.0, 0.0, 3.0, 0.0, -3.0, 0.0], dtype=np.float32)

        # Previous distance to goal, used for dense progress reward.
        self._prev_dist = None

    def _sample_noise(self) -> np.ndarray:
        if self.gaussian_noise:
            w = np.zeros(6)
            w[1] = np.random.normal(0, self.noise_std)
            w[3] = np.random.normal(0, self.noise_std)
            w[5] = np.random.normal(0, self.noise_std)
        else:
            w = np.zeros(6)
            for i in [1, 3, 5]:
                if np.random.rand() < 0.5:
                    w[i] = np.random.normal(0, self.noise_std)
                else:
                    w[i] = np.random.normal(0, 2.0 * self.noise_std)
        return w

    def _check_obstacle_collision(self, pos: np.ndarray) -> bool:
        x, y, z = pos
        for x_min, x_max, y_min, y_max, z_min, z_max in self.obstacles:
            if (x_min <= x <= x_max and
                y_min <= y <= y_max and
                z_min <= z <= z_max):
                return True
        return False

    def _check_goal(self, pos: np.ndarray) -> bool:
        return bool(np.all(pos >= self.goal_min) and np.all(pos <= self.goal_max))

    def _outside_corridor(self, pos: np.ndarray) -> bool:
        x, y, z = pos
        return (
            x < -15.0 or x > 15.0 or
            y < -9.0 or y > 9.0 or
            z < -7.0 or z > 7.0
        )

    def reset(
        self,
        random_start: bool = False,
        curriculum_frac: float | None = None,
    ) -> np.ndarray:
        """
        Reset the environment.

        Args:
            random_start: If True and curriculum_frac is None, sample a random
                valid start state (small jitter around canonical).
            curriculum_frac: If provided in [0, 1], interpolates the start
                state from an "easy" position next to the goal (frac=0) to the
                canonical far-away start (frac=1). A small jitter is added so
                the policy sees some state variation.
        """
        if curriculum_frac is not None:
            # Sample uniformly inside a box around the goal centre whose
            # half-width grows with ``curriculum_frac``. At frac=0 the box is
            # just slightly larger than the goal region; at frac=1 it covers
            # the entire corridor so that hard starts (including positions
            # near the canonical start) are possible. Obstacle positions,
            # corridor-exterior positions and in-goal positions are rejected.
            frac = float(np.clip(curriculum_frac, 0.0, 1.0))
            r_min = 3.0   # slightly larger than goal half-extents (2, 2, 2)
            r_max = 28.0  # covers canonical start distance from goal centre
            r = r_min + frac * (r_max - r_min)
            vel_jitter = 0.1 + 0.4 * frac

            x_lo = max(-15.0, self.goal_center[0] - r)
            x_hi = min(15.0, self.goal_center[0] + r)
            y_lo = max(-9.0, self.goal_center[1] - r)
            y_hi = min(9.0, self.goal_center[1] + r)
            z_lo = max(-7.0, self.goal_center[2] - r)
            z_hi = min(7.0, self.goal_center[2] + r)

            for _ in range(500):
                x = np.random.uniform(x_lo, x_hi)
                y = np.random.uniform(y_lo, y_hi)
                z = np.random.uniform(z_lo, z_hi)
                pos = np.array([x, y, z], dtype=np.float32)
                if self._check_obstacle_collision(pos):
                    continue
                if self._check_goal(pos):
                    # Trivial starts inside the goal give no policy gradient.
                    continue
                vx = np.random.normal(0.0, vel_jitter)
                vy = np.random.normal(0.0, vel_jitter)
                vz = np.random.normal(0.0, vel_jitter)
                self._state = np.array([x, vx, y, vy, z, vz], dtype=np.float32)
                break
            else:
                self._state = self.start_state.copy()
        elif not random_start:
            self._state = self.start_state.copy()
        else:
            # Sample a random start that is (a) not in an obstacle, (b) away
            # enough from the corridor boundary that a single small noise step
            # cannot push it outside, and (c) uses small initial velocities.
            for _ in range(1000):
                x = np.random.uniform(-14.0, 14.0)
                y = np.random.uniform(-8.0, 8.0)
                z = np.random.uniform(-6.5, 6.5)
                # Small initial velocities so the first step does not
                # deterministically leave the corridor.
                vx = np.random.normal(0.0, 0.3)
                vy = np.random.normal(0.0, 0.3)
                vz = np.random.normal(0.0, 0.3)
                state = np.array([x, vx, y, vy, z, vz], dtype=np.float32)
                pos = np.array([x, y, z], dtype=np.float32)
                if self._outside_corridor(pos):
                    continue
                if self._check_obstacle_collision(pos):
                    continue
                self._state = state
                break
            else:
                self._state = self.start_state.copy()

        self._done = False
        self._step_count = 0
        pos = np.array([self._state[0], self._state[2], self._state[4]], dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(pos - self.goal_center))
        return self._state.copy()

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        action = np.clip(action, self.u_min, self.u_max)
        old_state = self._state.copy()

        w = self._sample_noise()
        self._state = self.A @ self._state + self.B @ action + w

        self._state[1] = np.clip(self._state[1], -7.0, 7.0)
        self._state[3] = np.clip(self._state[3], -7.0, 7.0)
        self._state[5] = np.clip(self._state[5], -7.0, 7.0)

        pos = np.array([self._state[0], self._state[2], self._state[4]])
        self._step_count += 1

        new_dist = float(np.linalg.norm(pos - self.goal_center))
        progress = (self._prev_dist if self._prev_dist is not None else new_dist) - new_dist
        self._prev_dist = new_dist

        if self._outside_corridor(pos):
            reward = -5.0
            self._done = True
        elif self._check_obstacle_collision(pos):
            reward = -5.0
            self._done = True
        elif self._check_goal(pos):
            reward = 10.0
            self._done = True
        elif self._step_count >= self.max_steps:
            # Strong shaping-free terminal: distance-based penalty so that
            # timing out far from the goal is clearly worse than timing out
            # close to it.
            reward = -0.05 * new_dist
            self._done = True
        else:
            # Dense progress reward - scaled large enough that several good
            # steps offset a single step in the wrong direction. No per-step
            # "alive" cost: with sparse terminal penalties, a time cost can
            # push the policy toward deliberate self-termination when it
            # cannot yet see the goal.
            reward = 0.5 * progress - 0.01

        new_state = self._state.copy()
        return old_state, reward, new_state

    def is_done(self) -> bool:
        return self._done


if __name__ == '__main__':
    print("Testing UAV3D (JAIR 2022 benchmark):")
    env_3d = UAV3D(gaussian_noise=True)
    state = env_3d.reset()
    print(f"Initial state: {state}")
    print(f"  Position: [{state[0]:.1f}, {state[2]:.1f}, {state[4]:.1f}]")
    print(f"  Velocity: [{state[1]:.1f}, {state[3]:.1f}, {state[5]:.1f}]")
    print(f"Goal region: x in [{env_3d.goal_min[0]}, {env_3d.goal_max[0]}], "
          f"y in [{env_3d.goal_min[1]}, {env_3d.goal_max[1]}], "
          f"z in [{env_3d.goal_min[2]}, {env_3d.goal_max[2]}]")
    print(f"Number of obstacles: {len(env_3d.obstacles)}")

    for i in range(5):
        action = np.array([2.0, 0.5, -0.5])
        old_state, reward, new_state = env_3d.step(action)
        pos = np.array([new_state[0], new_state[2], new_state[4]])
        vel = np.array([new_state[1], new_state[3], new_state[5]])
        print(f"Step {i+1}: pos={pos}, vel={vel}, reward={reward:.3f}")

    print("\nCurriculum reset (frac=0, easy):")
    state = env_3d.reset(curriculum_frac=0.0)
    print(f"  Position: [{state[0]:.2f}, {state[2]:.2f}, {state[4]:.2f}]")
    print("Curriculum reset (frac=1, hard):")
    state = env_3d.reset(curriculum_frac=1.0)
    print(f"  Position: [{state[0]:.2f}, {state[2]:.2f}, {state[4]:.2f}]")
