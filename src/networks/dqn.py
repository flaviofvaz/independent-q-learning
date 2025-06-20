from flax import nnx
import jax.numpy as jnp
import jax
from abc import ABC


class Agent(ABC):
    def act(
        self,
    ):
        pass


class DQNAgent(Agent):
    def __init__(
        self, network: nnx.Module, action_space_dim: int, gamma: float, epsilon: float
    ):
        self.network = network
        self.epsilon = epsilon
        self.epsilon_decay_reate = 0.0000045
        self.epsilon_mininum = 0.1
        self.epsilon_evaluation = 0.05
        self.action_space_dim = action_space_dim


    def act(self, state: jnp.array, key: jax.random.PRNGKey, training=True) -> tuple[list[int], jax.random.PRNGKey]:
        """
        Selects actions for a batch of states using an epsilon-greedy policy.
        """
        if training:
            actions, key = _act_epsilon_greedy(self.network, state, self.epsilon, key)
            # adjust epsilon
            self.epsilon = max(self.epsilon_mininum, self.epsilon - self.epsilon_decay_reate)
        else:
            actions, key = _act_epsilon_greedy(self.network, state, self.epsilon_evaluation, key)
        return actions.tolist(), key


class DenseQNetwork(nnx.Module):
    def __init__(self, input_dimension, output_dimension, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(*input_dimension, 128, kernel_init=nnx.initializers.lecun_uniform(), rngs=rngs)
        self.linear2 = nnx.Linear(128, 128, kernel_init=nnx.initializers.lecun_uniform(), rngs=rngs)
        self.linear3 = nnx.Linear(128, output_dimension, kernel_init=nnx.initializers.lecun_uniform(), rngs=rngs)

    def __call__(self, x, training: bool = True):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
@nnx.jit
def _act_epsilon_greedy(network: nnx.Module, state: jnp.array, epsilon:jnp.array, key: jax.random.PRNGKey):
    num_agents = jnp.shape(state)[0]
    key, explore_key, action_key = jax.random.split(key, 3)

    # get q-values from the network for all agents
    q_values = network(state)
    greedy_actions = jnp.argmax(q_values, axis=1)

    # generate random actions for all agents
    random_actions = jax.random.randint(
        action_key,
        shape=(num_agents,),
        minval=0,
        maxval=6,
    )

    # decide for each agent whether to explore or exploit
    explore_decisions = jax.random.uniform(explore_key, shape=(num_agents,)) < epsilon

    # select actions based on the exploration decision
    actions = jnp.where(explore_decisions, random_actions, greedy_actions)

    return actions, key