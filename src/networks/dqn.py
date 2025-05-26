from flax import nnx
import jax.numpy as jnp
import jax
from abc import ABC


class Agent(ABC):
    def loss(self,):
        pass
    def act(self,):
        pass

class DQNAgent(Agent):
    def __init__(self, network: nnx.Module, action_space_dim: int, gamma: float, epsilon: float):
        self.network = network 
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_dim = action_space_dim 

    def act(self, state: jnp.array, key: jax.random.PRNGKey):
        new_key, subkey_uniform = jax.random.split(key)
        if jax.random.uniform(subkey_uniform) < self.epsilon:
            new_key, subkey_randint = jax.random.split(new_key)
            action = jax.random.randint(subkey_randint, shape=jnp.shape(state)[0], minval=0, maxval=self.action_space_dim)
        else:
            q_values = self.network(state)
            action = jnp.argmax(q_values, axis=1)
        return action.tolist(), new_key
    

class QNetwork(nnx.Module):
    def __init__(self, input_channels, output_dimension, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(input_channels, 32, kernel_size=(8,8), strides=(4,4), padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(4,4), strides=(2,2), padding="VALID", rngs=rngs)
        self.conv3 = nnx.Conv(64, 64, kernel_size=(3,3), strides=(1,1), padding="VALID", rngs=rngs)
        self.linear1 = nnx.Linear(3136, 512, rngs=rngs)
        self.linear2 = nnx.Linear(512, output_dimension, rngs=rngs)

    def __call__(self, x, training: bool=True):
      x = nnx.relu(self.conv1(x))
      x = nnx.relu(self.conv2(x))
      x = nnx.relu(self.conv3(x))
      x = x.reshape(x.shape[0], -1)
      x = self.linear1(x)
      out = self.linear2(x)
      return out
    
class DenseQNetwork(nnx.Module):
    def __init__(self, input_dimension, output_dimension, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(*input_dimension, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 256, rngs=rngs)
        self.linear3 = nnx.Linear(256, output_dimension, rngs=rngs)

    def __call__(self, x, training: bool=True):
      x = nnx.relu(self.linear1(x))
      x = nnx.relu(self.linear2(x))
      out = self.linear3(x)
      return out
