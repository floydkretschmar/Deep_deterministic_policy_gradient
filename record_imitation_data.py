import gym
import torch
import numpy as np
import Parameters
import laser_hockey_env as lh
from replay_memory import EfficientReplayMemory
from normalization import NormalizeActionWrapper, SkipStepWrapper


def main(env):
    player1 = lh.BasicOpponent()
    player2 = lh.BasicOpponent()

    state_size =env.observation_space.shape[0]
    action_size = env.action_space.shape[0] // 2

    buffer = EfficientReplayMemory(Parameters.IMITATION_BUFFER_SIZE, state_size, action_size)

    while len(buffer) < Parameters.IMITATION_BUFFER_SIZE:
        state = env.reset()
        obs_agent2 = env.obs_agent_two()
        while True:
            #env.render()
            action = player1.act(state)
            #a2 = player2.act(obs_agent2)
            a2 = [0,0,0]
            next_state, reward, done, info = env.step(np.hstack([action,a2]))  
            reward = 100 * reward + 50 * info["reward_closeness_to_puck"] + 100 * info["reward_touch_puck"] + 80 * info["reward_puck_direction"]

            """ if done and info["winner"] == 0:
                reward -= 5 """

            # build transition
            action = torch.Tensor([action])
            mask = torch.Tensor([not done])
            reward = torch.Tensor([reward])

            buffer.push(torch.Tensor([state]), action, reward, torch.Tensor([next_state]), mask)
            #buffer.push(torch.Tensor([state]), action, mask, torch.Tensor([next_state]), reward)

            obs_agent2 = env.obs_agent_two()
            if done: 
                break
            else:
                state = next_state



    buffer.save_memory("imitations_normal.pt")
    print("Saved imitation data")

if __name__ == '__main__':
    main(SkipStepWrapper(NormalizeActionWrapper(lh.LaserHockeyEnv(lh.LaserHockeyEnv.NORMAL)), Parameters.FRAME_SKIP))
