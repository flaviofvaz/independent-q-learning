import lbforaging

import gymnasium as gym
from typing import Tuple, List


class LbfEnvironment:
    def __init__(self, environment_name: str, grayscale: bool=True):
        self._env = gym.make(environment_name)
        
    def step(self, action: List[int]) -> Tuple:
        return self._env.step(action)
    
    def reset(self) -> Tuple:
        return self._env.reset()
    
    def close(self) -> None:
        self._env.close()

    def get_action_space(self) -> int:
        return self._env.action_space.n
    
    def sample_action(self,) -> int:
        return self._env.action_space.sample()
