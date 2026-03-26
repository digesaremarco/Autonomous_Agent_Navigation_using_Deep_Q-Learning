import numpy as np
from src import config

class QLearningTabular:
    def __init__(self, env):
        self.env = env

        self.gamma = config.GAMMA
        self.alpha = config.ALPHA

        self.epsilon = config.EPSILON_START
        self.epsilon_start = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay_steps = config.EPSILON_DECAY_STEPS

        # Q-table 7D
        self.Q = np.zeros(
            (
                config.NX,
                config.NY,
                config.N_THETA,
                8,   # goal direction bins
                3,   # sensor front
                3,   # sensor left
                3,   # sensor right
                config.N_ACTIONS
            ),
            dtype=np.float32
        )

        self.total_steps = 0

    def build_state(self, raw_state):
        '''
        Build enriched discrete state (tabular-friendly version of DQN state)
        '''

        x, y, theta = raw_state
        goal_x, goal_y = config.GOAL_POS

        # --- goal direction ---
        dx = goal_x - x
        dy = goal_y - y

        angle_to_goal = np.arctan2(dy, dx)
        theta_rad = theta * config.DELTA_THETA_RAD

        angle_diff = angle_to_goal - theta_rad

        # discretize into 8 bins
        igoal = int(((angle_diff + np.pi) / (2 * np.pi)) * 8) % 8

        # --- sensors ---
        sensors = self.env.get_sensors((x, y, theta))

        def discretize_sensor(s):
            if s < 0.2:
                return 0
            elif s < 0.5:
                return 1
            else:
                return 2

        s0 = discretize_sensor(sensors[0])
        s1 = discretize_sensor(sensors[1])
        s2 = discretize_sensor(sensors[2])

        return (x, y, theta, igoal, s0, s1, s2)

    def select_action(self, state):
        '''
        Epsilon-greedy action selection
        '''

        x, y, theta, igoal, s0, s1, s2 = state

        if np.random.rand() < self.epsilon:
            return np.random.choice(config.N_ACTIONS)
        else:
            return np.argmax(self.Q[x, y, theta, igoal, s0, s1, s2])



    def update_epsilon_episode(self, episode):
        fraction = min(episode / config.N_EPISODES, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def update_epsilon(self, episode):
        """
        Linear decay of epsilon over time
        """
        epsilon_decay = (config.EPSILON_START - config.EPSILON_END) / config.N_EPISODES
        self.epsilon = max(config.EPSILON_END, self.epsilon - epsilon_decay)

    def train(self):
        """
        Q-learning training loop with tracking of goals and collisions
        """
        goals_reached = 0
        collisions_count = 0

        for episode in range(1, config.N_EPISODES + 1):

            # --- Sample a valid initial state ---
            while True:
                x = np.random.randint(0, config.NX)
                y = np.random.randint(0, config.NY)
                theta = np.random.randint(0, config.N_THETA)

                raw_state = (int(x), int(y), int(theta))

                if not self.env.is_collision(raw_state):
                    break

            state = self.build_state(raw_state)

            episode_reward = 0.0
            done = False

            # --- Episode loop ---
            for step in range(config.MAX_STEPS_PER_EPISODE):

                # Select action
                action = self.select_action(state)

                # Environment step
                next_raw, reward, done = self.env.step(raw_state, action)
                next_raw = tuple(map(int, next_raw))

                next_state = self.build_state(next_raw)

                episode_reward += reward

                x, y, theta, igoal, s0, s1, s2 = state
                nx, ny, ntheta, nigoal, ns0, ns1, ns2 = next_state

                # --- Q-learning update ---
                best_next_q = 0.0 if done else np.max(
                    self.Q[nx, ny, ntheta, nigoal, ns0, ns1, ns2]
                )

                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[x, y, theta, igoal, s0, s1, s2, action]

                self.Q[x, y, theta, igoal, s0, s1, s2, action] += self.alpha * td_error

                # Move to next state
                raw_state = next_raw
                state = next_state
                self.total_steps += 1

                # Update epsilon
                self.update_epsilon(self.total_steps)

                if done:
                    if reward == config.R_GOAL:
                        goals_reached += 1
                    elif reward == config.R_COLLISION:
                        collisions_count += 1
                    break

            # --- Logging ---
            if episode % 100 == 0:
                success_rate = (goals_reached / episode) * 100
                print(
                    f"Episode {episode:5d}/{config.N_EPISODES} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Success Rate: {success_rate:5.1f}% ({goals_reached:5d} goals) | "
                    f"Collisions: {collisions_count:5d}"
                )

            self.update_epsilon_episode(episode)

        print(f"\nTraining Complete!")
        print(f"Final Statistics - Goals: {goals_reached}, Collisions: {collisions_count}")
        print(f"Final Success Rate: {(goals_reached / config.N_EPISODES) * 100:.2f}%")

    def get_greedy_action(self, raw_state):
        '''
        Get greedy action from enriched state
        '''

        state = self.build_state(raw_state)
        x, y, theta, igoal, s0, s1, s2 = state

        return np.argmax(self.Q[x, y, theta, igoal, s0, s1, s2])

    def evaluate(self, n_episodes=100):
        '''
        Evaluate learned policy
        '''

        success_count = 0

        for episode in range(n_episodes):
            raw_state = (0, 0, 0)
            done = False

            for step in range(config.MAX_STEPS_PER_EPISODE):
                action = self.get_greedy_action(raw_state)
                next_raw, reward, done = self.env.step(raw_state, action)

                raw_state = next_raw

                if done:
                    if self.env.is_goal(raw_state):
                        success_count += 1
                    break

        success_rate = success_count / n_episodes
        print(f"Evaluation: Success Rate = {success_rate:.2%} ({success_count}/{n_episodes})")

    def save_model(self, filename='q_learning_model.npz'):
        np.savez_compressed(filename, Q=self.Q)
        print(f"Model saved to {filename}")

    def load_model(self, filename='q_learning_model.npz'):
        try:
            data = np.load(filename)
            self.Q = data['Q']
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found. Starting with empty Q-table.")
            return False
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
            return False