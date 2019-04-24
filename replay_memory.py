import random
import torch
import numpy as np
from os.path import isfile

class EfficientReplayMemory(object):
    def __init__(self, capacity, state_size, action_size):
        self.max_capacity = capacity
        self.position = -1
        self.current_capacity = 0

        self.state = torch.zeros((capacity, state_size)).float()
        self.action = torch.zeros((capacity, action_size)).float()
        self.reward = torch.zeros((capacity, 1)).float()
        self.mask = torch.zeros((capacity, 1)).float()
        self.next_state = torch.zeros((capacity, state_size)).float()


    def push(self, state, action, reward, next_state, mask):
        """Saves a transition."""
        if self.current_capacity < self.max_capacity:
            self.current_capacity += 1
            self.position += 1
        else:
            self.position = (self.position + 1) % self.max_capacity

        self.state[self.position] = state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.mask[self.position] = mask

    def sample(self, batch_size):
        indices = [random.randint(0, self.current_capacity - 1) for _ in range(batch_size)]        
        return self.state[indices], self.action[indices], self.reward[indices], self.next_state[indices], self.mask[indices]

    def __len__(self):
        return self.current_capacity

    def save_memory(self, name):
        self.state.numpy().dump("state_imitations.dat")
        self.action.numpy().dump("action_imitations.dat")
        self.reward.numpy().dump("reward_imitations.dat")
        self.next_state.numpy().dump("next_state_imitations.dat")
        self.mask.numpy().dump("mask_imitations.dat")

    def load_memory(self, name):
        state_im = "state_" + name
        if isfile("state_imitations.dat"):
            self.state = torch.from_numpy(np.load("state_imitations.dat"))
            self.action = torch.from_numpy(np.load("action_imitations.dat"))
            self.reward = torch.from_numpy(np.load("reward_imitations.dat"))
            self.next_state = torch.from_numpy(np.load("next_state_imitations.dat"))
            self.mask = torch.from_numpy(np.load("mask_imitations.dat"))

            self.current_capacity = self.max_capacity
        else:
            print("No buffer to load")

