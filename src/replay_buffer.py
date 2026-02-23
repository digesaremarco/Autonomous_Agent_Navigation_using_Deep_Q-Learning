import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        """
        capacity: max number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Save a transition in the buffer
        """
        # Convert to appropriate types
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = float(reward)
        done = bool(done)

        data = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Return a random sample of transitions from the buffer, converted to torch tensors
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        # Convert to torch tensors
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)