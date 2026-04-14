import numpy as np

LENGTH = 5
WIDTH = 5
MAX_TURBULENCE = 3.5
MAX_VELOCITY = 3


class WetChicken:
    # Implements the 2-dimensional discrete Wet Chicken benchmark from 'Efficient Uncertainty Propagation for
    # Reinforcement Learning with Limited Data' by Alexander Hans and Steffen Udluft

    def __init__(self, length, width, max_turbulence, max_velocity):
        self.length = length
        self.width = width
        self.max_turbulence = max_turbulence
        self.max_velocity = max_velocity
        self._state = np.zeros(
            2)  # Don't use this state outside of this class, it is an array. Instead use get_state_int!

    def _get_reward(self):
        return self._state[0]

    def _velocity(self):
        return self.max_velocity * self._state[1] / self.width

    def _turbulence(self):
        return np.random.uniform(-1, 1) * (self.max_turbulence - self._velocity())

    def reset(self):
        self._state = np.zeros(2)
        return self._state.copy()

    def get_state(self):
        return self._state.copy()

    def step(self, action):
        action = np.clip(action, -1, 1)  # Action is a 2D vector with values in [-1, 1]^2

        x_hat = self._state[0] + action[0] + self._velocity() + self._turbulence()
        y_hat = self._state[1] + action[1]

        old_state = self._state.copy()

        if x_hat >= self.length:
            self._state[0] = 0
        elif x_hat < 0:
            self._state[0] = 0
        else:
            self._state[0] = x_hat

        if x_hat >= self.length or y_hat < 0:
            self._state[1] = 0
        elif y_hat >= self.width:
            self._state[1] = self.width
        else:
            self._state[1] = y_hat

        new_state = self._state.copy()
        return old_state, self._get_reward(), new_state


if __name__ == '__main__':
    wet_chicken = WetChicken(5, 5, 3.5, 3)
    action = np.array([0, 0])
    print(wet_chicken.step(action))
    print(wet_chicken.step(action))
    print(wet_chicken.step(action))

