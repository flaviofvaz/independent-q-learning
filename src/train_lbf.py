from networks.dqn import DQNAgent, DenseQNetwork
from environments.lbf import LbfEnvironment
from flax import nnx
import optax
import jax.numpy as jnp
import jax
import time
from memory import ReplayMemory


def initialize_replay_memory(memory: ReplayMemory, environment: LbfEnvironment, starting_capacity: int):
    # initialize counter
    counter = 0
    
    # initialize environment
    observation, _ = environment.reset()
    while counter < starting_capacity:
        action = environment.sample_action()
        new_observation, reward, terminated, truncated, info = environment.step(action)
        for i in range(len(observation)):
            memory.update_memory(observation[i], action[i], reward[i], new_observation[i], terminated or truncated)
            counter += 1
        if not terminated:
            observation = new_observation
        else:
            observation, _ = environment.reset()

        if counter % 1000 == 0:
            print(f"{memory.get_size()} entries added to memory")

def td_loss(model: nnx.Module, state: jnp.array, action: jnp.array, reward: jnp.array, next_state: jnp.array, is_done: jnp.array):
    # td estimate
    q_values = model(state)
    q_values = q_values[jnp.arange(0, jnp.shape(state)[0]), action.squeeze()]
    td_estimate = q_values
    
    # td target 
    q_values = model(next_state)
    max_q_values = jnp.max(q_values, axis=-1)
    td_target = reward.squeeze() + 0.99 * (1 - jnp.asarray(is_done.squeeze(), dtype=jnp.float32)) * max_q_values 

    td_error = jnp.pow(td_target - td_estimate, 2)
    return jnp.average(td_error)

@nnx.jit
def train_step(model: nnx.Module, state: jnp.array, action: jnp.array, reward: jnp.array,
               next_state: jnp.array, is_done: jnp.array, optimizer: nnx.Optimizer, 
               metrics: nnx.MultiMetric):
    
    grad_fn = nnx.value_and_grad(td_loss)
    loss, grads = grad_fn(model, state, action, reward, next_state, is_done)
    metrics.update(loss=loss)
    optimizer.update(grads)

def train():
    environment_name = "Foraging-8x8-2p-1f-coop-v3"
    rng = 0
    batch_size = 32
    frames = 10_000_000
    learning_rate = 0.0005
    memory_capacity = 100_000
    starting_capacity = 10_000
    action_space_dim = 6
    observation_space = (9,)
    train_every_n_steps = 4

    # for cpu development
    jax.config.update("jax_platforms", "cpu")

    # create environment
    environment = LbfEnvironment(environment_name)

    # initialize main random key
    key = jax.random.PRNGKey(rng)

    # initialize experience/replay memory
    print("Initializing experience memory")
    experience_memory = ReplayMemory(
        capacity=memory_capacity, 
        observation_shape=observation_space, 
        action_shape=1,
        rewards_shape=1, 
        dones_shape=1)

    # splitting random keys
    key, subkey = jax.random.split(key)

    # create network
    q_network = DenseQNetwork(observation_space, action_space_dim, rngs=nnx.Rngs(params=subkey) )
    
    # create dqn agent
    dqn_agent = DQNAgent(
            network=q_network,
            action_space_dim=action_space_dim,
            gamma=0.99,
            epsilon=1.0
        )

    # define optimizer
    optimizer = nnx.Optimizer(dqn_agent.network, optax.adam(learning_rate))

    # define metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    # initialize memory
    initialize_replay_memory(experience_memory, environment, starting_capacity)

    # initialize metrics history
    metrics_history = {'train_loss': []}

    print("Starting training")
    episodes_rewards = []
    current_reward = 0.0
    episode_counter = 0
    average_reward = 0.0

    start_time = time.time()
    state, _ = environment.reset()
    for i in range(frames):
        # get agents actions        
        agent_action, key = dqn_agent.act(state, key) 

        # perform actions
        new_state, reward, terminated, truncated, info = environment.step(agent_action)

        # update memory and episode rewards
        for j in range(len(state)):
            experience_memory.update_memory(state[j], agent_action[j], reward[j], new_state[j], terminated or truncated)
            current_reward += reward[j]

        # check if end of game
        if not terminated:
            state = new_state
        else:
            state, _ = environment.reset()
            episodes_rewards.append(float(current_reward))
            average_reward += float(current_reward)
            current_reward = 0.0
            episode_counter += 1

            if episode_counter % 100 == 0:
                print(
                    f"total episodes: {episode_counter}, "
                    f"frames seen: {i+1}, "
                    f"last 100 episodes average total reward: {average_reward / 100.0}, "
                    f"exploration rate: {dqn_agent.epsilon}, "
                    f"loss: {metrics_history['train_loss'][-1]}, "
                    f"total time: {time.time() - start_time}"
                )
                average_reward = 0.0
                start_time = time.time()

        if (i+1) % train_every_n_steps == 0:
            # sample batch of experiences
            batch, key = experience_memory.retrieve_experience(batch_size=batch_size, key=key)

            # get entries 
            state_batch = batch["state"]
            action_batch = batch["action"]
            reward_batch = batch["reward"]
            next_state_batch = batch["next_state"]
            is_done_batch = batch["is_done"]
        
            train_step(dqn_agent.network, state_batch, action_batch, reward_batch, next_state_batch, is_done_batch, optimizer, metrics)

            # Log training metrics
            for metric, value in metrics.compute().items():  
                metrics_history[f'train_{metric}'].append(value)  
                metrics.reset()  
        

if __name__=="__main__":
    train()
