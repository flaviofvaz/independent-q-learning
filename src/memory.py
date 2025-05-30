import jax.numpy as jnp
import jax
from typing import Tuple, List
from functools import partial


class ReplayMemory:
    def __init__(
        self, capacity, observation_shape, action_shape, rewards_shape, dones_shape
    ):
        self._capacity = capacity
        self._pointer = 0
        self._size = 0

        # Pre-allocate JAX arrays for efficiency
        self._states = jnp.zeros((capacity, *observation_shape), dtype=jnp.uint8)
        self._actions = jnp.zeros((capacity, action_shape), dtype=jnp.uint8)
        self._rewards = jnp.zeros((capacity, rewards_shape), dtype=jnp.float32)
        self._next_states = jnp.zeros((capacity, *observation_shape), dtype=jnp.uint8)
        self._dones = jnp.zeros((capacity, dones_shape), dtype=jnp.bool_)

    def update_memory(
        self,
        state: jnp.array,
        action: jnp.array,
        reward: jnp.array,
        next_state: jnp.array,
        is_done: jnp.array,
    ):
        # convert dtypes
        state_jax = jnp.array(state, dtype=jnp.uint8)
        action_jax = jnp.array(action, dtype=jnp.uint8)
        reward_jax = jnp.array(reward, jnp.float32)
        next_state_jax = jnp.array(next_state, dtype=jnp.uint8)
        done_jax = jnp.array(is_done, dtype=jnp.bool_)

        new_states, new_actions, new_rewards, new_next_states, new_dones, new_pointer, new_size = _update_memory(self._states, self._actions, self._rewards, self._next_states, self._dones, self._pointer, self._size, self._capacity,
                       state_jax, action_jax, reward_jax, next_state_jax, done_jax)

        self._states = new_states 
        self._actions = new_actions
        self._rewards = new_rewards
        self._next_states = new_next_states
        self._dones = new_dones
        self._pointer = int(new_pointer)
        self._size = int(new_size)
    

    def batch_update_memory(
        self,
        state: jnp.array,
        action: jnp.array,
        reward: jnp.array,
        next_state: jnp.array,
        is_done: jnp.array,
    ):
        # convert dtypes
        state_jax = jnp.asarray(state, dtype=jnp.uint8)
        action_jax = jnp.expand_dims(jnp.asarray(action, dtype=jnp.uint8), axis=-1)
        reward_jax = jnp.expand_dims(jnp.asarray(reward, jnp.float32), axis=-1)
        next_state_jax = jnp.asarray(next_state, dtype=jnp.uint8)
        done_jax = jnp.expand_dims(jnp.asarray(is_done, dtype=jnp.bool_), axis=-1)

        new_states, new_actions, new_rewards, new_next_states, new_dones, new_pointer, new_size = _batch_update_memory(self._states, self._actions, self._rewards, self._next_states, self._dones, self._pointer, self._size, self._capacity,
                       state_jax, action_jax, reward_jax, next_state_jax, done_jax)

        self._states = new_states 
        self._actions = new_actions
        self._rewards = new_rewards
        self._next_states = new_next_states
        self._dones = new_dones
        self._pointer = int(new_pointer)
        self._size = int(new_size)


    def retrieve_experience(self, batch_size: int, key: jax.random.PRNGKey):
        # split random key
        new_key, subkey = jax.random.split(key)

        # generate random indices
        indices = jax.random.randint(
            key=subkey, shape=(batch_size,), minval=0, maxval=self._size
        )

        random_batch = _retrieve_from_memory(self._states, self._actions, self._rewards, self._next_states, self._dones, indices)
        return random_batch, new_key

    def get_capacity(
        self,
    ):
        return self._capacity

    def get_size(
        self,
    ):
        return self._size
    

#@partial(jax.jit, static_argnums=(5, 6))
@jax.jit
def _retrieve_from_memory(states, actions, rewards, next_states, dones, indices):
    # # split random key
    # new_key, subkey = jax.random.split(key)

    # # generate random indices
    # indices = jax.random.randint(
    #     key=subkey, shape=(batch_size,), minval=0, maxval=memory_size
    # )

    # retrieve a batch of random entries
    random_batch = {
        "state": states[indices],
        "action": actions[indices],
        "reward": rewards[indices],
        "next_state": next_states[indices],
        "is_terminal": dones[indices],
    }
    return random_batch


@jax.jit
def _update_memory(
    states, actions, rewards, next_states, dones, pointer, size, capacity,
    state_new, action_new, reward_new, next_state_new, is_done_new
):
    states = states.at[pointer].set(state_new)
    actions = actions.at[pointer].set(action_new)
    rewards = rewards.at[pointer].set(reward_new)
    next_states = next_states.at[pointer].set(next_state_new)
    dones = dones.at[pointer].set(is_done_new)

    # update pointer and size
    new_pointer = (pointer + 1) % capacity
    new_size = jnp.minimum(size + 1, capacity)

    return states, actions, rewards, next_states, dones, new_pointer, new_size


@jax.jit
def _batch_update_memory(
    states, actions, rewards, next_states, dones, pointer, size, capacity,
    state_new, action_new, reward_new, next_state_new, is_done_new
):
    batch_size = state_new.shape[0]
    indices = (pointer + jnp.arange(batch_size)) % capacity

    states = states.at[indices].set(state_new)
    actions = actions.at[indices].set(action_new)
    rewards = rewards.at[indices].set(reward_new)
    next_states = next_states.at[indices].set(next_state_new)
    dones = dones.at[indices].set(is_done_new)

    # update pointer and size
    new_pointer = (pointer + batch_size) % capacity
    new_size = jnp.minimum(size + batch_size, capacity)

    return states, actions, rewards, next_states, dones, new_pointer, new_size
