from networks.dqn import DQNAgent, DenseQNetwork
from environments.lbf import LbfEnvironment
from flax import nnx
import optax
import jax.numpy as jnp
import jax
import time
from memory import ReplayMemory


# for cpu development
jax.config.update("jax_platforms", "cpu")

def initialize_replay_memory(
    memory: ReplayMemory, environment: LbfEnvironment, starting_capacity: int
):
    # initialize counter
    counter = 0

    # initialize environment
    observation, _ = environment.reset()

    obss = []
    actionss = []
    new_obss = []
    rewardss = []
    doness = []
    while counter < starting_capacity:
        action = environment.sample_action()
        new_observation, reward, terminated, truncated, _ = environment.step(action)
        data_len = len(new_observation)
        done_flag = terminated or truncated
        
        obss.extend(observation)
        actionss.extend(action)
        new_obss.extend(new_observation)
        rewardss.extend(reward)
        doness.extend([done_flag] * len(observation))
        counter += data_len

        if len(obss) % 1000 == 0:
            memory.batch_update_memory(
                obss,
                actionss,
                rewardss,
                new_obss,
                doness,
            )
            obss = []
            actionss = []
            new_obss = []
            rewardss = []
            doness = []

        if not terminated:
            observation = new_observation
        else:
            observation, _ = environment.reset()

        if counter % 1000 == 0:
            print(f"{memory.get_size()} entries added to memory")

@nnx.jit
def td_loss(
    model: nnx.Module,
    target_model: nnx.Module,
    state: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_state: jnp.array,
    is_done: jnp.array,
):
    # td estimate
    q_values = model(state)
    action_squeezed = action.squeeze(axis=-1) 
    td_estimate = q_values[jnp.arange(q_values.shape[0]), action_squeezed]

    # td target
    target_q_values = jax.lax.stop_gradient(target_model(next_state))
    max_q_values = jnp.max(target_q_values, axis=-1)
    td_target = reward.squeeze(-1) + 0.99 * (1.0 - is_done.squeeze(-1)) * max_q_values

    # td error
    delta = td_target - td_estimate
    huber_loss = jnp.where(jnp.abs(delta) < 1.0, 
                            0.5 * delta**2, 
                            jnp.abs(delta) - 0.5)
    return jnp.mean(huber_loss)

@nnx.jit
def train_step(
    model: nnx.Module,
    target_model: nnx.Module,
    state: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_state: jnp.array,
    is_done: jnp.array,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
):

    grad_fn = nnx.value_and_grad(td_loss)
    loss, grads = grad_fn(model, target_model, state, action, reward, next_state, is_done)
    metrics.update(loss=loss)
    optimizer.update(grads)


def train():
    environment_name = "Foraging-8x8-2p-2f-coop-v3"
    rng = 0
    batch_size = 32
    frames = 10_000_000
    learning_rate = 0.0003
    memory_capacity = 1_000_000
    starting_capacity = 50_000
    action_space_dim = 6
    observation_space = (12,)
    train_every_n_steps = 4
    update_target_every_n_steps = 10_000
    evaluate_every_n_steps = 100_000
    evaluation_episodes = 100
    evaluation_epsilon = 0.05

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
        dones_shape=1,
    )
    
    # splitting random keys
    key, subkey = jax.random.split(key)

    # create network
    q_network = DenseQNetwork(
        observation_space, action_space_dim, rngs=nnx.Rngs(params=subkey)
    )

    # create target network
    nnx_graphdef, nnx_state = nnx.split(q_network)
    target_q_network = nnx.merge(nnx_graphdef, nnx_state)

    # create dqn agent
    dqn_agent = DQNAgent(
        network=q_network, action_space_dim=action_space_dim, gamma=0.99, epsilon=1.0
    )

    # define optimizer
    optimizer = nnx.Optimizer(dqn_agent.network, optax.adam(learning_rate))

    # define metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # initialize memory
    initialize_replay_memory(experience_memory, environment, starting_capacity)

    # initialize metrics history
    metrics_history = {"train_loss": []}

    print("Starting training")
    episodes_rewards = []
    current_reward = 0.0
    episode_counter = 0
    average_reward = 0.0

    obss = []
    actionss = []
    new_obss = []
    rewardss = []
    doness = []

    start_time = time.time()
    state, _ = environment.reset()
    for i in range(frames):
        state = jnp.array(state, dtype=jnp.float32)
        # get agents actions
        agent_action, key = dqn_agent.act(state, key)

        # perform actions
        new_state, reward, terminated, truncated, _ = environment.step(agent_action)

        done_flag = terminated or truncated
        dones = jnp.full((2,), done_flag, dtype=jnp.bool_)
        
        obss.extend(state)
        actionss.extend(agent_action)
        new_obss.extend(new_state)
        rewardss.extend(reward)
        doness.extend(dones)
        current_reward += sum(reward)

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

        if (i + 1) % train_every_n_steps == 0:
            # update memory and episode rewards
            experience_memory.batch_update_memory(
                obss,
                actionss,
                rewardss,
                new_obss,
                doness,
            )
            
            obss = []
            actionss = []
            new_obss = []
            rewardss = []
            doness = []

            key, subkey = jax.random.split(key)
            # sample batch of experiences
            batch, key = experience_memory.retrieve_experience(
                batch_size=batch_size, key=key
            )

            state_batch = jnp.array(batch["state"], dtype=jnp.float32)
            action_batch = batch["action"]
            reward_batch = batch["reward"]
            next_state_batch = jnp.array(batch["next_state"], dtype=jnp.float32)
            is_terminal_batch = jnp.array(batch["is_terminal"], dtype=jnp.float32)

            train_step(
                dqn_agent.network,
                target_q_network,
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                is_terminal_batch,
                optimizer,
                metrics,
            )

            # Log training metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
                metrics.reset()
        
        if (i + 1) % update_target_every_n_steps == 0:
            # update target network
            nnx_graphdef, nnx_state = nnx.split(dqn_agent.network)
            target_q_network = nnx.merge(nnx_graphdef, nnx_state)

        if (i + 1) % evaluate_every_n_steps == 0:
            total_rewards = 0.0
            total_foods_collected = 0.0
            episodes_completed = 0
            # evaluate on N episodes
            for _ in range(evaluation_episodes):
                curr_reward, foods_collected, key = run_episode(environment, dqn_agent, key)
                total_rewards += curr_reward
                total_foods_collected += float(foods_collected / 2.0)
                if foods_collected == 2:
                    episodes_completed += 1
            total_rewards /= float(evaluation_episodes)
            total_foods_collected /= float(evaluation_episodes)
            print(f"Step {i+1} - Evaluation - Average rewards: {total_rewards} - Average Food collection: {total_foods_collected} - Episodes completed: {episodes_completed}")
            state, _ = environment.reset()
            current_reward = 0.0


def run_episode(environment, dqn_agent, key):
    done = False
    state, _ = environment.reset()
    foods_collected = 0
    total_reward = 0.0
    while not done:
        state = jnp.asarray(state, dtype=jnp.float32)
        agent_action, key = dqn_agent.act(state, key, False)
        new_state, reward, terminated, truncated, _ = environment.step(agent_action)
        curr_reward = sum(reward)
        if curr_reward > 0:
            foods_collected += 1
        total_reward += curr_reward
        state = new_state
        done = terminated or truncated
    return total_reward, foods_collected, key


if __name__ == "__main__":
    train()
