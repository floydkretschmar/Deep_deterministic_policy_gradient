import torch


class QFunction(torch.nn.Module):
    """
    Class used to judge the predicted value of our actor. In the case
    of DDPG we are approximating Q(s,a).
    """

    def __init__(self, state_size, action_size):
        """
        :param state_size: Dimension of input state
        :param action_size: Dimension of output action
        """
        super(QFunction, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.state_network = torch.nn.Sequential(torch.nn.Linear(self.state_size, 256),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(256, 128),
                                                 torch.nn.ReLU())

        self.action_network = torch.nn.Sequential(torch.nn.Linear(self.action_size, 128),
                                                  torch.nn.ReLU())

        self.q_network = torch.nn.Sequential(torch.nn.Linear(256, 128),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(128, 1))

        self.state_network.to(self.device)
        self.action_network.to(self.device)
        self.q_network.to(self.device)
        
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.normal_(m.bias.data)

    def forward(self, state, action):
        """
        Returns the Q value for the given state and action.
        :param state: The state for which to approximate the Q-Value [batch_size,state_size]
        :param action: The action for which to approximate the Q-Value [batch_size,action_size]
        :return: Q(state, action) [batch_size,1]
        """
        state_out = self.state_network(state.to(self.device))
        action_out = self.action_network(action.to(self.device))
        q = self.q_network(torch.cat((state_out, action_out), dim=1))

        return q.cpu()
