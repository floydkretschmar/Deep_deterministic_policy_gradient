import numpy as np

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_size, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_size = action_size
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_size) * self.mu

	def reset(self):
		self.X = np.ones(self.action_size) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X