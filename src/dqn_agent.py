import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

from src import config
from src.network import QNetwork
from src.replay_buffer import ReplayBuffer



class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        buffer_capacity=300_000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        tau=0.001,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=250_000
    ):

        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        self.loss_fn = nn.SmoothL1Loss() # Huber loss

    def select_action(self, state):
        '''
        Epsilon-greedy action selection
        '''

        self.total_steps += 1

        # Epsilon decay
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.total_steps / self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return q_values.argmax(dim=1).item()


    def store(self, state, action, reward, next_state, done):
        '''
        Store a transition in the replay buffer
        '''
        self.replay_buffer.push(state, action, reward, next_state, done)


    def train_step(self):
        '''
        Perform a single training step of the DQN algorithm
        '''

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        state, action, reward, next_state, done = \
            self.replay_buffer.sample(self.batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Current Q values
        current_q = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():

            # Double DQN: action selection from q_net, evaluation from target_net
            next_actions = self.q_net(next_state).argmax(1)
            next_q = self.target_net(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = reward + self.gamma * next_q * (1 - done)

        # Loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0) # Gradient clipping to prevent exploding gradients
        self.optimizer.step()

        # Soft update target network
        self.soft_update()

        return loss.item()


    def soft_update(self):
        '''
        Soft update of target network parameters
        '''
        for target_param, param in zip(
            self.target_net.parameters(),
            self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def sample_start_state(self, env, episode):
        '''
        Sample a valid start state for the episode, optionally using curriculum learning
        '''
        goal_x, goal_y = config.GOAL_POS

        if not config.USE_CURRICULUM:
            # random start state sampling without curriculum
            while True:
                x = np.random.randint(0, config.NX)
                y = np.random.randint(0, config.NY)
                theta = np.random.randint(0, config.N_THETA)

                if not env.is_collision((x, y, theta)):
                    return x, y, theta

        # curriculum learning
        max_dist = None
        for start_ep, end_ep, dist in config.CURRICULUM_PHASES:
            if start_ep <= episode < end_ep:
                max_dist = dist
                break

        while True:
            if max_dist is None:
                # fase hard → random sampling in the whole grid
                x = np.random.randint(0, config.NX)
                y = np.random.randint(0, config.NY)
            else:
                # fase easy → sample near the goal
                dx = np.random.randint(-max_dist, max_dist + 1)
                dy = np.random.randint(-max_dist, max_dist + 1)

                x = np.clip(goal_x + dx, 0, config.NX - 1)
                y = np.clip(goal_y + dy, 0, config.NY - 1)

            theta = np.random.randint(0, config.N_THETA)

            if not env.is_collision((x, y, theta)):
                return x, y, theta

    def train(self, env, num_episodes=2000, max_steps=500):
        '''
        Main training loop for DQN
        :param env: the environment to train on
        :param num_episodes: total number of episodes to train for
        :param max_steps: maximum steps per episode to prevent infinite loops
        '''

        episode_rewards = []
        episode_losses = []

        for episode in range(num_episodes):

            x, y, theta = self.sample_start_state(env, episode)

            theta_rad = theta * config.DELTA_THETA_RAD
            goal_x, goal_y = config.GOAL_POS
            sensors = env.get_sensors((x, y, theta))

            state = np.array([
                x / config.NX,
                y / config.NY,
                np.sin(theta_rad),
                np.cos(theta_rad),
                (goal_x - x) / config.NX,
                (goal_y - y) / config.NY,
                sensors[0],
                sensors[1],
                sensors[2]
            ], dtype=np.float32)

            total_reward = 0
            total_loss = 0
            steps = 0

            for step in range(max_steps):

                action = self.select_action(state)

                next_raw, reward, done = env.step(
                    (x, y, theta),
                    action,
                    continuous=True
                )

                nx, ny, ntheta = next_raw

                ntheta_rad = ntheta * config.DELTA_THETA_RAD
                sensors = env.get_sensors((nx, ny, ntheta))
                next_state = np.array([
                    nx / config.NX,
                    ny / config.NY,
                    np.sin(ntheta_rad),
                    np.cos(ntheta_rad),
                    (goal_x - nx) / config.NX,
                    (goal_y - ny) / config.NY,
                    sensors[0],
                    sensors[1],
                    sensors[2]
                ], dtype=np.float32)


                self.store(state, action, reward, next_state, done)
                loss = self.train_step()

                total_reward += reward
                total_loss += loss
                steps += 1

                state = next_state
                x, y, theta = nx, ny, ntheta

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_losses.append(total_loss / max(1, steps))

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(episode_losses[-100:])

                print("--------------------------------------------------")
                print(f"Episode: {episode}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Loss   (last 100): {avg_loss:.5f}")
                print(f"Epsilon: {self.epsilon:.4f}")

    def extract_policy(self, env):
        """
        Extract the learned policy from the Q-network
        by taking the argmax action for each discrete state.
        """

        policy = np.zeros(
            (config.NX, config.NY, config.N_THETA),
            dtype=int
        )

        self.q_net.eval()
        goal_x, goal_y = config.GOAL_POS

        total_states = config.NX * config.NY * config.N_THETA

        with torch.no_grad():
            with tqdm(total=total_states, desc="Extracting policy", unit="state") as pbar:
                for x in range(config.NX):
                    for y in range(config.NY):
                        for theta in range(config.N_THETA):
                            theta_rad = theta * config.DELTA_THETA_RAD

                            sensors = env.get_sensors((x, y, theta))

                            state = np.array([
                                x / config.NX,
                                y / config.NY,
                                np.sin(theta_rad),
                                np.cos(theta_rad),
                                (goal_x - x) / config.NX,
                                (goal_y - y) / config.NY,
                                sensors[0],  # front
                                sensors[1],  # left
                                sensors[2]  # right
                            ], dtype=np.float32)

                            state_t = torch.tensor(
                                state, dtype=torch.float32
                            ).unsqueeze(0).to(self.device)

                            q_values = self.q_net(state_t)
                            best_action = q_values.argmax(dim=1).item()

                            policy[x, y, theta] = best_action
                            pbar.update(1)

        print("Policy extraction completed.", flush=True)
        return policy

    def greedy_action(self, env, state_triplet):
        """
        Get the greedy action for a given state by querying the Q-network
        """
        x, y, theta = state_triplet

        theta_rad = theta * config.DELTA_THETA_RAD
        goal_x, goal_y = config.GOAL_POS
        sensors = env.get_sensors((x, y, theta))

        state = np.array([
            x / config.NX,
            y / config.NY,
            np.sin(theta_rad),
            np.cos(theta_rad),
            (goal_x - x) / config.NX,
            (goal_y - y) / config.NY,
            sensors[0],
            sensors[1],
            sensors[2]
        ], dtype=np.float32)

        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = q_values.argmax(dim=1).item()

        return action

    def save_model(self, path="dqn_model.pth"):
        '''
        Save the Q-network weights to a file
        '''
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="dqn_model.pth"):
        '''
        Load the Q-network weights from a file
        '''
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"Model loaded from {path}")
        else:
            print(f"No model file found at {path}. Starting with random weights.")