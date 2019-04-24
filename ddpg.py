import sys
from os.path import isfile

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import numpy as np
from math import sqrt

from replay_memory import EfficientReplayMemory
from noise import OrnsteinUhlenbeckActionNoise
from critic import QFunction
from actor import Policy
import Parameters

class DDPGAgent(object):
    def __init__(self, state_size, action_size, action_bound_high, action_bound_low, imitation_data_path):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_bound_high = torch.Tensor([action_bound_high]).to(device)
        self.action_bound_low = torch.Tensor([action_bound_low]).to(device)
        self.action_size = action_size

        self.buffer = EfficientReplayMemory(Parameters.BUFFER_SIZE, self.state_size, self.action_size)
        self.imitation_buffer = EfficientReplayMemory(Parameters.IMITATION_BUFFER_SIZE, self.state_size, self.action_size)
        self.imitation_buffer.load_memory(imitation_data_path)
        self.imitation_lambda = Parameters.IMITATION_LAMBDA

        # Actor
        self.policy_function = Policy(self.state_size, self.action_size, self.action_bound_high)
        self.policy_function_target = Policy(self.state_size, self.action_size, self.action_bound_high)
        self.policy_function_noisy = Policy(self.state_size, self.action_size, self.action_bound_high)
        self.policy_function_optim = Adam(self.policy_function.parameters(), lr=Parameters.ACTOR_LEARNING_RATE)
        self.imitation_optimizer = Adam(self.policy_function.parameters(), lr=self.imitation_lambda)

        # critic 1 (q-value)
        self.q_function = QFunction(self.state_size, self.action_size)
        self.q_function_target = QFunction(self.state_size, self.action_size)
        self.q_function_optim = Adam(self.q_function.parameters(), lr=Parameters.CRITIC_LEARNING_RATE)

        # Noise parameters
        self.action_noise = OrnsteinUhlenbeckActionNoise(self.action_size)
        self.desired_action_std = Parameters.DESIRED_ACTION_STD
        self.current_noise_std = Parameters.INITIAL_NOISE_STD
        self.coefficient = Parameters.ADAPT_COEFFICIENT

        # hyperparameters
        self.gamma = Parameters.GAMMA
        self.tau = Parameters.TAU

        self.hard_update_network(self.policy_function_target, self.policy_function)
        self.hard_update_network(self.q_function_target, self.q_function)

    def soft_update_network(self, target, source):
        for target_parameters, source_parameters in zip(target.parameters(), source.parameters()):
            target_parameters.data.copy_(target_parameters.data * (1.0 - self.tau) + source_parameters.data * self.tau)

    def hard_update_network(self, target, source):
        target.load_state_dict(source.state_dict())

    def chose_action(self, state, exploration=True):
        self.policy_function.eval()

        if exploration and Parameters.PARAMETER_NOISE:
            action = self.policy_function_noisy((Variable(state)))
        else:
            action = self.policy_function((Variable(state)))

        self.policy_function.train()
        action = action.data

        if self.action_noise is not None and exploration:
            action += torch.Tensor(self.action_noise.sample())

        return action.clamp(-1, 1)

    def store_buffer_transition(self, state, action, mask, next_state, reward):
        self.buffer.push(state, action, reward, next_state, mask)

    def smooth_l1_loss(self, input, target, beta=1, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch because current pytorch variant is buggy
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

    def train(self):
        # sample batch and train
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(Parameters.BATCH_SIZE)
        loss_imitation = 0

        state_batch = Variable(state_batch)
        action_batch = Variable(action_batch)
        reward_batch = Variable(reward_batch)
        mask_batch = Variable(mask_batch)
        next_state_batch = Variable(next_state_batch)
        
        # train the critic (Q-function)
        next_actions = self.policy_function_target(next_state_batch)
        next_q_values = self.q_function_target(next_state_batch, next_actions)
        expected_q_values = reward_batch + (self.gamma * mask_batch * next_q_values)

        self.q_function_optim.zero_grad()
        predicted_q_values = self.q_function(state_batch, action_batch)
        #q_value_loss = F.smooth_l1_loss(predicted_q_values, expected_q_values)
        #q_value_loss = F.smooth_l1_loss(expected_q_values, predicted_q_values)
        q_value_loss = self.smooth_l1_loss(expected_q_values, predicted_q_values)
        #q_value_loss = (predicted_q_values - expected_q_values).pow(2).mean()
        q_value_loss.backward()
        self.q_function_optim.step()

        # train the policy 
        self.policy_function_optim.zero_grad()

        q_value_prediction = self.q_function(state_batch,self.policy_function(state_batch))

        # maximize the Q value for the chosen action
        policy_loss = - q_value_prediction
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_function_optim.step()

        if Parameters.USE_IMITATION_LEARNING:
            state_batch_imitation, action_batch_imitation, _, _, _ = self.imitation_buffer.sample(Parameters.IMITATION_BATCH_SIZE)

            action_batch_imitation = Variable(action_batch_imitation, requires_grad=True)
            state_batch_imitation = Variable(state_batch_imitation, requires_grad=True)
            predicted_actions = self.chose_action(state_batch_imitation, False)
            q_value_prediction = self.q_function(state_batch_imitation,predicted_actions)
            q_value_imitation = self.q_function(state_batch_imitation,action_batch_imitation)

            # Only try to learn the actions that were actually better than the current policy
            imitation_mask = (q_value_imitation > q_value_prediction)
            
            self.imitation_optimizer.zero_grad()

            loss_imitation = ((predicted_actions - action_batch_imitation) * imitation_mask.float()).pow(2).mean()
            loss_imitation.backward()

            self.imitation_optimizer.step()

        # update the target networks
        self.update_networks()

        return q_value_loss.item(), policy_loss.item()

    def update_networks(self):
        self.soft_update_network(self.policy_function_target, self.policy_function)
        self.soft_update_network(self.q_function_target, self.q_function)
        
    
    def noise_actor_parameters(self):
        """
        Apply dynamic noise to the actor network PARAMETERS for better exploration.
        See:
        https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
        https://blog.openai.com/better-exploration-with-parameter-noise/
        """
        self.hard_update_network(self.policy_function_noisy, self.policy_function)
        params = self.policy_function_noisy.state_dict()
        for key in params:
            if 'ln' in key: 
                pass 
            param = params[key]
            param += (torch.randn(param.shape) * self.current_noise_std).to(self.policy_function_noisy.device)

    def adapt_parameter_noise(self, states, actions):
        """
        Adapt the rate of noise dynamically according to a specified target.
        See:
        https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
        https://blog.openai.com/better-exploration-with-parameter-noise/
        """
        states = torch.cat(states, 0)
        unperturbed_actions = self.chose_action(states, False)
        perturbed_actions = torch.cat(actions, 0)

        # calculate euclidian distance of both actions:
        mean_diff = np.mean(np.square((perturbed_actions-unperturbed_actions).numpy()), axis=0)
        distance = sqrt(np.mean(mean_diff))

        # adapt the standard deviation of the parameter noise
        if distance > self.desired_action_std:
            self.current_noise_std /= self.coefficient
        else:
            self.current_noise_std *= self.coefficient
    
    def save_models(self, path="./"):
        torch.save(self.policy_function.state_dict(), path + "actor.pt")
        torch.save(self.q_function.state_dict(), path + "critic.pt")
        print("Models saved successfully")

    def load_models(self, path="./"):
        if isfile(path + "actor.pt"):
            self.policy_function.load_state_dict(torch.load(path + "actor.pt"))
            self.q_function.load_state_dict(torch.load("critic.pt"))
            self.policy_function_target.load_state_dict(self.policy_function.state_dict())
            self.q_function_target.load_state_dict(self.q_function.state_dict())
            print("Models loaded succesfully")
        else:
            print("No model to load")