import gym
import laser_hockey_env as lh

class SkipStepWrapper(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipStepWrapper, self).__init__(env)
        self.skip = skip

    def obs_agent_two(self):
        return self.env.obs_agent_two()
    
    def _step(self, action):
        total_reward = 0
        total_puck_closeness = 0
        total_touch = 0
        total_direction = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if isinstance(self.env, lh.LaserHockeyEnv):
                total_puck_closeness += info["reward_closeness_to_puck"]
                total_touch += info["reward_touch_puck"] 
                total_direction += info["reward_puck_direction"]
            info['steps'] = i + 1
            if done:
                break
        info["reward_closeness_to_puck"] = total_puck_closeness
        info["reward_touch_puck"] = total_touch
        info["reward_puck_direction"] = total_direction
        return obs, total_reward, done, info


class NormalizeActionWrapper(gym.ActionWrapper):
    def obs_agent_two(self):
        return self.env.obs_agent_two()

    def action(self, action):
        action = (action + 1) / 2 
        action *= (self.action_space.high[0] - self.action_space.low[0])
        action += self.action_space.low[0]
        return action

    def reverse_action(self, action):
        action -= self.action_space.low[0]
        action /= (self.action_space.high[0] - self.action_space.low[0])
        action = action * 2 - 1
        return action
