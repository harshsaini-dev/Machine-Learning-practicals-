# Lab Sheet 05 - Harsh Saini

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run this file or notebook from the practicals folder.
DATA = Path("datasets")
OUT = Path("outputs") / "lab_05"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 78)
pd.set_option("display.max_columns", 8)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style="whitegrid", font_scale=0.9)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=160)
    plt.show()

import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from collections import deque
import random
import time

torch.set_num_threads(1)
# CPU runs with fixed seeds; timing and minor results may vary.
SEED = 42


# Program 01: Install and configure the Gymnasium (or OpenAI Gym) library.

print("Gymnasium version:", gym.__version__)
print("Install command: python -m pip install gymnasium")
env = gym.make("FrozenLake-v1", is_slippery=False,
               render_mode="ansi")
state, info = env.reset(seed=SEED)
print("FrozenLake created; start state:", state)


# Program 02: Create and execute a simple Reinforcement Learning environment.

state, info = env.reset(seed=SEED)
print("Initial state:", state)
print(env.render())
next_state, reward, terminated, truncated, info = env.step(2)
print("After moving right:", next_state)
print("Reward:", reward)
print("Terminated:", terminated, "Truncated:", truncated)


# Program 03: Explore the observation space and action space of an RL environment.

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
print("Actions: 0=left, 1=down, 2=right, 3=up")


# Program 04: Display the states, actions, rewards, and termination conditions of an environment.

print("State IDs:\n", np.arange(16).reshape(4, 4))
print("Map:\n", env.unwrapped.desc.astype("U1"))
print("S=start, F=frozen, H=hole, G=goal")
print("Reward: 1 at goal; otherwise 0.")
print("Termination: goal or hole.")
print("Truncation: the time limit is reached.")


# Program 05: Simulate random actions in the FrozenLake environment.

env.action_space.seed(SEED)
state, _ = env.reset(seed=SEED)
trajectory = []
for step in range(20):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    trajectory.append([step + 1, state, action, next_state, reward])
    state = next_state
    if terminated or truncated:
        break
print(pd.DataFrame(trajectory,
    columns=["step", "state", "action", "next_state", "reward"]))


# Program 06: Implement the Q-Learning algorithm for the FrozenLake environment.

def train_q(environment, episodes=3000, alpha=0.5,
            gamma=0.95, epsilon=0.8, decay=True, seed=42):
    rng = np.random.default_rng(seed)
    q = np.zeros((environment.observation_space.n,
                  environment.action_space.n))
    rewards = []
    changes = []
    started = time.perf_counter()
    steps = 0
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        total_reward = 0.0
        max_change = 0.0
        explore = (max(0.05, epsilon * (0.997 ** episode))
                   if decay else epsilon)
        while True:
            if rng.random() < explore:
                action = int(rng.integers(environment.action_space.n))
            else:
                best = np.flatnonzero(q[state] == q[state].max())
                action = int(rng.choice(best))
            next_state, reward, terminated, truncated, _ = (
                environment.step(action))
            # A true terminal state has no future value.
            future = 0.0 if terminated else q[next_state].max()
            old_value = q[state, action]
            q[state, action] += alpha * (
                reward + gamma * future - old_value)
            max_change = max(max_change,
                             abs(q[state, action] - old_value))
            state = next_state
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        changes.append(max_change)
    return {"q": q, "rewards": np.array(rewards),
            "changes": np.array(changes), "steps": steps,
            "seconds": time.perf_counter() - started}

def evaluate_q(environment, q, episodes=100, seed=9000):
    rewards = []
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.array(rewards)

print("Q-learning functions defined.")
print("Q <- Q + alpha * (reward + gamma * max_next_Q - Q)")


# Program 07: Initialize and update the Q-table during training.

example_q = np.zeros((16, 4))
state, action, reward, next_state = 14, 2, 1.0, 15
alpha, gamma = 0.5, 0.95
terminal = True
future = 0.0 if terminal else example_q[next_state].max()
example_q[state, action] += alpha * (
    reward + gamma * future - example_q[state, action])
print("Q-table shape:", example_q.shape)
print("Updated Q[14, 2]:", example_q[14, 2])


# Program 08: Train the agent for multiple episodes using Q-Learning.

frozen = train_q(env, episodes=3000, seed=SEED)
q_table = frozen["q"]
print("Training episodes:", len(frozen["rewards"]))
print("Total environment steps:", frozen["steps"])
print("Goal visits:", int(frozen["rewards"].sum()))
print("Last 100 training mean reward:",
      round(frozen["rewards"][-100:].mean(), 3))


# Program 09: Display the learned Q-table after training.

print(pd.DataFrame(q_table,
    columns=["left", "down", "right", "up"]).round(3))
np.save(OUT / "frozenlake_q.npy", q_table)


# Program 10: Evaluate the trained Q-Learning agent.

frozen_eval = evaluate_q(env, q_table)
print("Evaluation episodes:", len(frozen_eval))
print("Success rate:", round(float((frozen_eval > 0).mean()), 3))
print("Mean reward:", round(frozen_eval.mean(), 3))


# Program 11: Plot the cumulative rewards obtained during training.

plt.figure(figsize=(6, 3))
plt.plot(np.cumsum(frozen["rewards"]))
plt.xlabel("Training episode")
plt.ylabel("Cumulative reward (goal visits)")
plt.title("FrozenLake Q-learning")
save_plot("11_cumulative_rewards")


# Program 12: Study the effect of different learning rates on Q-Learning performance.

def q_sweep(parameter, values):
    rows = []
    for value in values:
        training_means, evaluation_means = [], []
        for seed in [42, 43, 44]:
            arguments = {parameter: value}
            if parameter == "epsilon":
                arguments["decay"] = False
            result = train_q(env, episodes=1000, seed=seed, **arguments)
            training_means.append(result["rewards"][-100:].mean())
            evaluation_means.append(evaluate_q(env, result["q"]).mean())
        rows.append({parameter: value,
            "train_last100": np.mean(training_means),
            "eval_mean": np.mean(evaluation_means),
            "eval_std": np.std(evaluation_means)})
    return pd.DataFrame(rows)

alpha_results = q_sweep("alpha", [0.1, 0.5, 0.9])
print(alpha_results)


# Program 13: Study the effect of different discount factor (Gamma) values.

gamma_results = q_sweep("gamma", [0.5, 0.9, 0.99])
print(gamma_results)


# Program 14: Compare exploration and exploitation using different epsilon values.

epsilon_results = q_sweep("epsilon", [0.1, 0.5, 0.9])
print(epsilon_results)


# Program 15: Implement an epsilon-greedy action selection policy.

def epsilon_greedy(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return int(np.argmax(q[state]))

rng = np.random.default_rng(SEED)
state = 0
print("Greedy action:", int(np.argmax(q_table[state])))
print("Ten epsilon-greedy actions (epsilon=0.3):")
print([epsilon_greedy(q_table, state, 0.3, rng)
       for _ in range(10)])


# Program 16: Design a simple Grid World environment.

class GridWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(16)
        self.action_space = spaces.Discrete(4)
        self.walls = {5, 7, 11}
        self.goal = 15
        self.state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1, 2 or 3")
        if self.state == self.goal:
            raise RuntimeError("Reset after reaching the goal")
        row, column = divmod(self.state, 4)
        movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dr, dc = movement[action]
        new_row = int(np.clip(row + dr, 0, 3))
        new_column = int(np.clip(column + dc, 0, 3))
        candidate = new_row * 4 + new_column
        if candidate not in self.walls:
            self.state = candidate
        terminated = self.state == self.goal
        reward = 1.0 if terminated else -0.01
        return self.state, reward, terminated, False, {}

grid = gym.wrappers.TimeLimit(GridWorld(), max_episode_steps=64)
print("Grid:", grid.observation_space, grid.action_space)
print("Walls:", sorted(grid.unwrapped.walls))
print("Start:", grid.reset(seed=SEED)[0], "Goal:", grid.unwrapped.goal)


# Program 17: Train a Q-Learning agent in the custom Grid World.

grid_result = train_q(grid, episodes=2000, seed=SEED)
grid_eval = evaluate_q(grid, grid_result["q"])
print("Training episodes:", len(grid_result["rewards"]))
print("Evaluation mean reward:", round(grid_eval.mean(), 3))
print("Successful episodes:", int((grid_eval > 0).sum()))
np.save(OUT / "gridworld_q.npy", grid_result["q"])


# Program 18: Visualize the optimal path learned by the agent.

state, _ = grid.reset(seed=10000)
path = [state]
for _ in range(64):
    action = int(np.argmax(grid_result["q"][state]))
    state, reward, terminated, truncated, _ = grid.step(action)
    path.append(state)
    if terminated or truncated:
        break
print("Greedy learned path:", path)
print("Path length:", len(path) - 1)
print("Reached goal:", state == grid.unwrapped.goal)
# Verify against shortest-path length using breadth-first search.
queue = deque([(0, 0)])
visited = {0}
shortest = None
while queue:
    node, distance = queue.popleft()
    if node == 15:
        shortest = distance
        break
    row, column = divmod(node, 4)
    for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        r, c = row + dr, column + dc
        nxt = r * 4 + c
        if (0 <= r < 4 and 0 <= c < 4 and
                nxt not in visited and nxt not in grid.unwrapped.walls):
            visited.add(nxt)
            queue.append((nxt, distance + 1))
print("Shortest possible steps:", shortest)
plt.figure(figsize=(4, 4))
board = np.zeros((4, 4))
for wall in grid.unwrapped.walls:
    board[wall // 4, wall % 4] = 1
plt.imshow(board, cmap="Greys", vmin=0, vmax=1)
plt.plot([s % 4 for s in path], [s // 4 for s in path], "o-")
for s in range(16):
    plt.text(s % 4, s // 4, str(s), ha="center", va="bottom",
             color="red" if s in grid.unwrapped.walls else "black")
plt.xticks(range(4))
plt.yticks(range(4))
plt.title("Learned Grid World path")
save_plot("18_grid_path")


# Program 19: Compare agent performance in FrozenLake and Grid World environments.

print(pd.DataFrame({
    "environment": ["FrozenLake (non-slippery)", "Grid World"],
    "states": [16, 16],
    "actions": [4, 4],
    "success_rate": [(frozen_eval > 0).mean(),
                     (grid_eval > 0).mean()],
    "mean_return": [frozen_eval.mean(), grid_eval.mean()]}))
print("Reward rules differ; mean returns are not a fair ranking.")


# Program 20: Analyze the convergence behavior of the Q-Learning algorithm.

reward_average = pd.Series(frozen["rewards"]).rolling(100).mean()
change_average = pd.Series(frozen["changes"]).rolling(100).mean()
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(reward_average)
axes[0].set_title("Reward: 100-episode mean")
axes[0].set_xlabel("Episode")
axes[1].plot(change_average)
axes[1].set_title("Q update: 100-episode mean")
axes[1].set_xlabel("Episode")
save_plot("20_q_convergence")
print("Final 100-episode mean maximum Q update:",
      round(frozen["changes"][-100:].mean(), 6))


# Program 21: Install the required libraries for Deep Q-Network implementation.

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("Device: CPU")
print("Install: python -m pip install torch gymnasium matplotlib")
print("Tensor:", torch.tensor([1.0, 2.0, 3.0]))


# Program 22: Implement a basic Deep Q-Network using TensorFlow or PyTorch.

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, state):
        return self.layers(state)

def evaluate_dqn(model, episodes=20, seed=10000):
    environment = gym.make("CartPole-v1")
    scores = []
    model.eval()
    with torch.no_grad():
        for episode in range(episodes):
            state, _ = environment.reset(seed=seed + episode)
            total = 0.0
            while True:
                values = model(torch.tensor(state, dtype=torch.float32))
                action = int(values.argmax().item())
                state, reward, terminated, truncated, _ = environment.step(action)
                total += reward
                if terminated or truncated:
                    break
            scores.append(total)
    environment.close()
    return np.array(scores)

def train_dqn(episodes=250, seed=42, replay=True,
              use_target=True, learning_rate=0.001):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = gym.make("CartPole-v1")
    online, target = DQN(), DQN()
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    memory = deque(maxlen=10000)
    rewards = []
    steps = 0
    epsilon = 1.0
    gamma = 0.99
    batch_size = 64
    started = time.perf_counter()
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        total = 0.0
        while True:
            if random.random() < epsilon:
                action = random.randrange(2)
            else:
                with torch.no_grad():
                    values = online(torch.tensor(state, dtype=torch.float32))
                    action = int(values.argmax().item())
            next_state, reward, terminated, truncated, _ = environment.step(action)
            # Bootstrap at time-limit truncation, not true termination.
            transition = (state.copy(), action, reward,
                          next_state.copy(), float(terminated))
            memory.append(transition)
            steps += 1
            # Same update frequency and optimizer for all ablations.
            ready = len(memory) >= batch_size if replay else True
            if ready and steps % 4 == 0:
                batch = (random.sample(memory, batch_size) if replay
                         else [transition])
                states, actions, values, next_states, terminals = zip(*batch)
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                values = torch.tensor(values, dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                terminals = torch.tensor(terminals, dtype=torch.float32)
                predicted = online(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    network = target if use_target else online
                    future = network(next_states).max(dim=1).values
                    expected = values + gamma * (1 - terminals) * future
                loss = loss_function(predicted, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
            if use_target and steps % 250 == 0:
                target.load_state_dict(online.state_dict())
            state = next_state
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        epsilon = max(0.05, epsilon * 0.98)
    environment.close()
    return {"model": online, "rewards": np.array(rewards),
            "seconds": time.perf_counter() - started, "steps": steps,
            "seed": seed, "learning_rate": learning_rate}

print(DQN())
print("Inputs: four CartPole observations; outputs: two Q-values.")
print("Replay capacity: 10000; batch: 64; gamma: 0.99.")


# Program 23: Train a DQN agent on the CartPole environment.

dqn_result = train_dqn(episodes=250, seed=SEED)
print("Training episodes:", len(dqn_result["rewards"]))
print("Environment steps:", dqn_result["steps"])
print("First 20 mean reward:", dqn_result["rewards"][:20].mean())
print("Last 20 mean reward:", dqn_result["rewards"][-20:].mean())


# Program 24: Plot the episode-wise reward obtained during DQN training.

plt.figure(figsize=(6, 3))
plt.plot(dqn_result["rewards"], alpha=0.35, label="Episode reward")
plt.plot(pd.Series(dqn_result["rewards"]).rolling(20).mean(),
         label="20-episode mean")
plt.xlabel("Training episode")
plt.ylabel("CartPole return")
plt.legend()
save_plot("24_dqn_rewards")


# Program 25: Evaluate the trained DQN agent.

dqn_eval = evaluate_dqn(dqn_result["model"], episodes=20)
print("Greedy evaluation episodes:", len(dqn_eval))
print("Mean return:", round(dqn_eval.mean(), 3))
print("Standard deviation:", round(dqn_eval.std(), 3))
print("Minimum / maximum:", dqn_eval.min(), dqn_eval.max())
print("500-step episodes:", int((dqn_eval == 500).sum()))


# Program 26: Compare Q-Learning and DQN performance.

# A binned Q-table provides a same-environment comparison.
bins = [np.linspace(-2.4, 2.4, 7), np.linspace(-3, 3, 7),
        np.linspace(-0.21, 0.21, 9), np.linspace(-3, 3, 7)]
shape = tuple(len(edges) + 1 for edges in bins) + (2,)

def discretize(state):
    return tuple(int(np.digitize(value, edges))
                 for value, edges in zip(state, bins))

cart_env = gym.make("CartPole-v1")
cart_q = np.zeros(shape)
rng = np.random.default_rng(SEED)
cart_rewards = []
cart_steps = 0
started = time.perf_counter()
for episode in range(250):
    state, _ = cart_env.reset(seed=SEED + episode)
    epsilon = max(0.05, 0.98 ** episode)
    total = 0.0
    while True:
        key = discretize(state)
        action = (int(rng.integers(2)) if rng.random() < epsilon
                  else int(np.argmax(cart_q[key])))
        nxt, reward, terminated, truncated, _ = cart_env.step(action)
        future = 0 if terminated else cart_q[discretize(nxt)].max()
        cart_q[key + (action,)] += 0.2 * (
            reward + 0.99 * future - cart_q[key + (action,)])
        state = nxt
        total += reward
        cart_steps += 1
        if terminated or truncated:
            break
    cart_rewards.append(total)
cart_seconds = time.perf_counter() - started
cart_eval = []
for episode in range(20):
    state, _ = cart_env.reset(seed=10000 + episode)
    total = 0
    while True:
        action = int(np.argmax(cart_q[discretize(state)]))
        state, reward, terminated, truncated, _ = cart_env.step(action)
        total += reward
        if terminated or truncated:
            break
    cart_eval.append(total)
cart_env.close()
cart_eval = np.array(cart_eval)
print(pd.DataFrame({
    "algorithm": ["Binned Q-learning", "DQN"],
    "training_episodes": [250, 250],
    "environment_steps": [cart_steps, dqn_result["steps"]],
    "evaluation_mean": [cart_eval.mean(), dqn_eval.mean()],
    "evaluation_std": [cart_eval.std(), dqn_eval.std()]}))


# Program 27: Analyze the effect of replay memory on DQN performance.

# Two seeds per setting, same 250-episode training budget.
ablations = []
for name, replay, target_network in [
        ("Replay + target", True, True),
        ("No replay", False, True),
        ("No target", True, False)]:
    for seed in [42, 43]:
        run = train_dqn(episodes=250, seed=seed, replay=replay,
                        use_target=target_network)
        scores = evaluate_dqn(run["model"])
        ablations.append({"setting": name, "seed": seed,
                          "eval_mean": scores.mean(),
                          "last20_train": run["rewards"][-20:].mean(),
                          "steps": run["steps"]})
ablation_table = pd.DataFrame(ablations)
print(ablation_table.loc[
    ablation_table["setting"].isin(["Replay + target", "No replay"])])


# Program 28: Study the role of the target network in DQN.

print(ablation_table.loc[
    ablation_table["setting"].isin(["Replay + target", "No target"])])
print("Target network: frozen bootstrap estimates between copies.")
print("No target: bootstrap with the current online network.")


# Program 29: Save the trained DQN model.

torch.save({"state_dict": dqn_result["model"].state_dict(),
            "seed": SEED, "episodes": 250}, OUT / "cartpole_dqn.pt")
print("Saved: cartpole_dqn.pt")
print("Architecture: 4 -> 64 -> 64 -> 2")


# Program 30: Load the saved DQN model and perform testing.

checkpoint = torch.load(OUT / "cartpole_dqn.pt",
                        map_location="cpu", weights_only=True)
loaded_dqn = DQN()
loaded_dqn.load_state_dict(checkpoint["state_dict"])
loaded_scores = evaluate_dqn(loaded_dqn)
print("Reloaded evaluation mean:", loaded_scores.mean())
print("Matches original evaluation:",
      np.array_equal(loaded_scores, dqn_eval))


# Program 31: Compare cumulative rewards obtained using different RL algorithms.

print("CartPole total training rewards:")
print("Binned Q-learning:", np.sum(cart_rewards))
print("DQN:", dqn_result["rewards"].sum())
plt.figure(figsize=(6, 3))
plt.plot(np.cumsum(cart_rewards), label="Binned Q-learning")
plt.plot(np.cumsum(dqn_result["rewards"]), label="DQN")
plt.xlabel("Training episode")
plt.ylabel("Cumulative CartPole reward")
plt.legend()
save_plot("31_algorithm_cumulative")


# Program 32: Visualize the learning curve of the RL agent.

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(pd.Series(frozen["rewards"]).rolling(100).mean())
axes[0].set_title("FrozenLake Q-learning")
axes[0].set_ylabel("100-episode mean reward")
axes[1].plot(pd.Series(dqn_result["rewards"]).rolling(20).mean())
axes[1].set_title("CartPole DQN")
axes[1].set_ylabel("20-episode mean reward")
for axis in axes:
    axis.set_xlabel("Episode")
save_plot("32_learning_curves")


# Program 33: Compare training time and convergence of Q-Learning and DQN.

timing = pd.DataFrame({
    "algorithm": ["Binned Q-learning", "DQN"],
    "seconds": [cart_seconds, dqn_result["seconds"]],
    "training_steps": [cart_steps, dqn_result["steps"]],
    "last20_mean": [np.mean(cart_rewards[-20:]),
                    dqn_result["rewards"][-20:].mean()]})
timing["steps_per_second"] = timing["training_steps"] / timing["seconds"]
print(timing)
print("Timing includes interaction and training, excludes evaluation.")


# Program 34: Analyze the impact of hyperparameters on learning performance.

lr_rows = []
for rate in [0.0005, 0.002]:
    for seed in [42, 43]:
        run = train_dqn(episodes=250, seed=seed, learning_rate=rate)
        scores = evaluate_dqn(run["model"])
        lr_rows.append({"learning_rate": rate, "seed": seed,
                        "eval_mean": scores.mean()})
# The default-rate runs already exist in the ablation baseline.
for row in ablations:
    if row["setting"] == "Replay + target":
        lr_rows.append({"learning_rate": 0.001, "seed": row["seed"],
                        "eval_mean": row["eval_mean"]})
lr_table = pd.DataFrame(lr_rows).sort_values(["learning_rate", "seed"])
print("DQN learning-rate results (250 episodes):")
with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
    print(lr_table)
    print("Mean by rate:")
    print(lr_table.groupby("learning_rate")["eval_mean"].agg(["mean", "std"]))


# Program 35: Prepare a comparative report summarizing the performance of Reinforcement Learning algorithms.

report = pd.DataFrame({
    "algorithm": ["Q-learning", "Q-learning", "Binned Q-learning", "DQN"],
    "environment": ["FrozenLake", "Grid World", "CartPole", "CartPole"],
    "training_episodes": [3000, 2000, 250, 250],
    "eval_mean_return": [frozen_eval.mean(), grid_eval.mean(),
                         cart_eval.mean(), dqn_eval.mean()]})
print(report)
print("FrozenLake: non-slippery; reward 1 only at the goal.")
print("Grid World: goal reward 1 and a -0.01 step penalty.")
print("CartPole: reward 1 per step, maximum 500 steps.")
print("Tabular Q stores values; DQN approximates values with a network.")
print("The two CartPole runs use matching evaluation seeds.")
print("Use more training seeds before claiming one method is better.")
report.to_csv(OUT / "rl_comparison.csv", index=False)
ablation_table.to_csv(OUT / "dqn_ablations.csv", index=False)
lr_table.to_csv(OUT / "dqn_learning_rates.csv", index=False)
env.close()
grid.close()
