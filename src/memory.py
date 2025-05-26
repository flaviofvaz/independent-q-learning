import jax.numpy as jnp
import jax
from typing import Tuple, List


class ReplayMemory:
    def __init__(self, capacity, observation_shape, action_shape, rewards_shape, dones_shape):
        self._capacity = capacity
        self._pointer = 0
        self._size = 0

        # Pre-allocate JAX arrays for efficiency
        self._states = jnp.zeros((capacity, *observation_shape), dtype=jnp.int8)
        if isinstance(action_shape, Tuple) or isinstance(action_shape, List):
            self._actions = jnp.zeros((capacity, *action_shape), dtype=jnp.int8)
        else:
            self._actions = jnp.zeros((capacity, action_shape), dtype=jnp.int8)

        if isinstance(rewards_shape, Tuple) or isinstance(rewards_shape, List):
            self._rewards = jnp.zeros((capacity, *rewards_shape), dtype=jnp.float32)
        else:
            self._rewards = jnp.zeros((capacity, rewards_shape), dtype=jnp.float32)
        self._next_states = jnp.zeros((capacity, *observation_shape), dtype=jnp.int8)
        
        if isinstance(dones_shape, Tuple) or isinstance(dones_shape, List):
            self._dones = jnp.zeros((capacity, *dones_shape), dtype=jnp.bool_)
        else:
            self._dones = jnp.zeros((capacity, dones_shape), dtype=jnp.bool_)


    def update_memory(self, state: jnp.array, action: int, reward: float, next_state: jnp.array, is_done: bool):
        # convert dtypes
        state_jax = jnp.array(state, dtype=jnp.int8)
        action_jax = jnp.array(action, dtype=jnp.int8)
        reward_jax = jnp.array(reward, jnp.float32)
        next_state_jax = jnp.array(next_state, dtype=jnp.int8)
        done_jax = jnp.array(is_done, dtype=jnp.bool_)

        # insert in memory
        self._states = self._states.at[self._pointer].set(state_jax)
        self._actions = self._actions.at[self._pointer].set(action_jax)
        self._rewards = self._rewards.at[self._pointer].set(reward_jax)
        self._next_states = self._next_states.at[self._pointer].set(next_state_jax)
        self._dones = self._dones.at[self._pointer].set(done_jax)
        
        # update pointer and size
        self._pointer = (self._pointer + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def retrieve_experience(self, batch_size: int, key: jax.random.PRNGKey):
        # sanity check
        if self._size < batch_size:
            raise ValueError("Not enough experiences in memory to sample a batch.")
        
        # split random key
        new_key, subkey = jax.random.split(key)
        
        # generate random indices
        indices = jax.random.randint(key=subkey, shape=(batch_size,), minval=0, maxval=self._size)
        
        # retrieve a batch of random entries
        random_batch = {
            "state": self._states[indices],
            "action": self._actions[indices],
            "reward": self._rewards[indices],
            "next_state": self._next_states[indices],
            "is_done": self._dones[indices]
        }
        return random_batch, new_key
    
    def get_capacity(self,):
        return self._capacity
    
    def get_size(self,):
        return self._size