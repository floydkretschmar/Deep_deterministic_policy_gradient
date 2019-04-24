import torch
import Parameters


class Policy(torch.nn.Module):
    """
    Class used to predict an action. We are trying to approximate pi(a | s)
    """

    def __init__(self, state_size, action_size, action_bound):
        """
        :param state_size: Dimension of input state
        :param action_size: Dimension of output action
        :param action_bound: Defines the action space by defining [-lower_bound,upper_bound].
                                Approximated actions will be in this range.
        """
        super(Policy, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound

        self.network = torch.nn.Sequential(torch.nn.Linear(self.state_size, 256),
                                           torch.nn.LayerNorm(256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 128),
                                           torch.nn.LayerNorm(128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(
                                               128, self.state_size),
                                           torch.nn.LayerNorm(self.state_size),
                                           torch.nn.ReLU())

        self.output = torch.nn.Sequential(torch.nn.Linear(self.state_size, self.action_size),
                                          torch.nn.Tanh())

        self.network.to(self.device)
        self.output.to(self.device)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.normal_(m.bias.data)

    def normalize(self, layer_output):
        return layer_output

    def forward(self, state):
        """
        Returns an action for the given state according to the policy pi.
        :param state: Input state [n,state_size]
        :return: Chosen action [n,action_size]
        """
        state = state.to(self.device)
        action = self.network(state)

        # use residual connection
        if Parameters.USE_RESIDUALS:
            action = action + state

        #action = self.normalize(action)
        action = self.output(action)

        # scaling to action bound
        action = action * self.action_bound

        return action.cpu()
