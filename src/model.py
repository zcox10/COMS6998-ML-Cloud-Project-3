import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=4):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.block1 = nn.Linear(hidden_dim, hidden_dim)
        self.block2 = nn.Linear(hidden_dim, hidden_dim)
        self.block3 = nn.Linear(hidden_dim, hidden_dim)
        self.block4 = nn.Linear(hidden_dim, hidden_dim)
        self.block5 = nn.Linear(hidden_dim, hidden_dim)
        self.block6 = nn.Linear(hidden_dim, hidden_dim)
        self.block7 = nn.Linear(hidden_dim, hidden_dim)
        self.block8 = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input to hidden layer
        x = self.relu(self.input_layer(x))

        # residual block
        x1 = self.relu(self.block1(x))
        x2 = self.block2(x1)
        x = self.relu(x + x2)

        # residual block
        x3 = self.relu(self.block3(x))
        x4 = self.block4(x3)
        x = self.relu(x + x4)

        # residual block
        x5 = self.relu(self.block5(x))
        x6 = self.block6(x5)
        x = self.relu(x + x6)

        # residual block
        x7 = self.relu(self.block7(x))
        x8 = self.block8(x7)
        x = self.relu(x + x8)

        # output layer
        out = self.output_layer(x)
        return out
