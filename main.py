import matplotlib.pyplot as plt

import gym
import numpy as np
from gym import wrappers
import gc

import torch
from ddpg import DDPGAgent
from normalization import NormalizeActionWrapper, SkipStepWrapper
import Parameters
import laser_hockey_env as lh

import argparse

# game mode
NORMAL = lh.LaserHockeyEnv.NORMAL
TRAIN_SHOOTING = lh.LaserHockeyEnv.TRAIN_SHOOTING
TRAIN_DEFENSE = lh.LaserHockeyEnv.TRAIN_DEFENSE
PENDULUM = 4

def create_environment(mode, create_test_environment=False):
    if mode == PENDULUM:
        environment = gym.make("Pendulum-v0")
        action_size = environment.action_space.shape[0]
    else:
        if mode == NORMAL:
            environment = lh.LaserHockeyEnv()
        elif mode == TRAIN_SHOOTING:
            environment = lh.LaserHockeyEnv(lh.LaserHockeyEnv.TRAIN_SHOOTING)
        elif mode == TRAIN_DEFENSE:
            environment = lh.LaserHockeyEnv(lh.LaserHockeyEnv.TRAIN_DEFENSE)
        action_size = environment.action_space.shape[0] // 2

    if Parameters.SKIP_FRAMES and not create_test_environment:
        environment = SkipStepWrapper(environment, Parameters.FRAME_SKIP)
    
    if Parameters.NORMALIZE_ACTIONS:
        environment = NormalizeActionWrapper(environment)

    return environment, action_size


def test(environment, agent, mode, enemy_agent, render=False):
    state = torch.Tensor([environment.reset()])
    epoch_reward = 0
    while True:
        if render:
            environment.render()
        action = agent.chose_action(state, False)

        next_state, reward, done, _ = take_step(mode, environment, action, enemy_agent)
        epoch_reward += reward

        next_state = torch.Tensor([next_state])

        state = next_state
        if done:
            break

    return epoch_reward
        

def take_step(mode, environment, action, enemy=None):
    if mode == PENDULUM:
        next_state, reward, done, info = environment.step(action.numpy()[0])
    else:
        if enemy == None:
            action_enemy = [0,0.,0]
        else: 
            enemy_state = environment.obs_agent_two()
            action_enemy = enemy.act(enemy_state)
        
        next_state, reward, done, info = environment.step(np.hstack([action.numpy().reshape(-1,), action_enemy]))
        reward = 100 * reward + 50 * info["reward_closeness_to_puck"] + 100 * info["reward_touch_puck"] + 80 * info["reward_puck_direction"]
                    
    return next_state, reward, done, info

def train_batch(agent, learning_steps, rewards=None):
    steps = learning_steps
    for _ in range(Parameters.FITTING_ITERATIONS):
        value_loss, policy_loss = agent.train()
        steps += 1
    return steps, value_loss, policy_loss

def train(environment, agent, mode, enemy_agent=None):
    steps = 0
    learning_steps = 0
    train_steps = 0
    training_rewards = []
    test_rewards = []
    c_losses = []
    a_losses = []
    avg_rewards = []

    for epoch in range(Parameters.EPOCHS):
        state = torch.Tensor([environment.reset()])

        # reset action and parameter space noise
        if epoch % 20 == 0:
            agent.action_noise.reset()

        agent.noise_actor_parameters()

        states = []
        actions = []

        epoch_reward = 0
        episode_length = 0
        while True:
            action = agent.chose_action(state)
            next_state, reward, done, _ = take_step(mode, environment, action, enemy_agent)
            steps += 1
            epoch_reward += reward
            episode_length += 1

            # build transition
            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            # store transition
            agent.store_buffer_transition(state, action, mask, next_state, reward)
            states.append(state)
            actions.append(action)
            

            state = next_state

            if len(agent.buffer) > Parameters.BATCH_SIZE and Parameters.LEARN_ONLINE and steps > Parameters.WASHOUT:
                learning_steps, c_loss, a_loss = train_batch(agent, learning_steps)
                a_losses.append(-a_loss)
                c_losses.append(c_loss)
                train_steps += 1
            if done:
                break

        if not Parameters.LEARN_ONLINE and len(agent.buffer) > Parameters.BATCH_SIZE and steps > Parameters.WASHOUT:
            for _ in range(episode_length):
                learning_steps = train_batch(agent, learning_steps) 

        training_rewards.append(epoch_reward)

        # change the parameter space noise level
        agent.adapt_parameter_noise(states, actions)

        # run 5 testing runs every 50 epochs
        if epoch % 50 == 0:
            test_reward = 0
            for _ in range(5):
                test_reward += test(environment, agent, mode, enemy_agent)

            test_reward /= 5            
            test_rewards.append(test_reward)

            avg_reward = np.mean(training_rewards[-5:])
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(epoch, steps, test_reward, avg_reward))
            avg_rewards.append(avg_reward)


        gc.collect()

        if epoch % 500 == 0:
            agent.save_models(str(epoch))

    plt.plot(range(len(a_losses)), a_losses)
    plt.ylabel('actor score')
    plt.xlabel('training step')
    plt.savefig("actor_loss.png")
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(range(len(c_losses)), c_losses)
    plt.ylabel('critic loss')
    plt.xlabel('training step')
    plt.savefig("critic_loss.png")
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(range(0, Parameters.EPOCHS, 50), avg_rewards)
    plt.ylabel('average train reward (ws=5)')
    plt.xlabel('epoch')
    plt.savefig("avg_rewards.png")
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.plot(range(0, Parameters.EPOCHS, 50), test_rewards)
    plt.ylabel('test reward')
    plt.xlabel('epoch')
    plt.savefig("rewards.png")
    plt.clf()
    plt.cla()
    plt.close()

    agent.save_models()
    environment.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing', type=int, default=0)
    parser.add_argument('--mode', type=int, default=PENDULUM)
    args = parser.parse_args()

    mode = args.mode
    player2 = lh.BasicOpponent()

    if mode == TRAIN_SHOOTING:
        imitation_data = "imitations_shooting.pt"
    elif mode == TRAIN_DEFENSE:
        imitation_data = "imitations_defense.pt"
    else:
        imitation_data = "imitations_normal.pt"

    environment, action_size = create_environment(mode, args.testing)
    agent = DDPGAgent(environment.observation_space.shape[0], action_size, environment.action_space.high[0], environment.action_space.low[0], imitation_data)
    if args.testing:
        agent.load_models()
        for _ in range(20):
            test(environment, agent, mode, player2, True)
    else:
        #agent.load_models()
        train(environment, agent, mode, player2)

