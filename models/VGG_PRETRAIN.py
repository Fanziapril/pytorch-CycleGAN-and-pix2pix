import Torch

class VggEncoder(nn.Module):
    def __init__(self, config=config):
        super(VggEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))

        self.conv4 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))

        self.pool2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv5 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))

        self.conv6 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))

        self.conv7 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))

        self.pool3 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv8 = nn.Conv2d(256, 512, (3, 3), (1, 1), 1)

        self.conv9 = nn.Conv2d(512, 512, (3, 3), (1, 1), 1)

        self.conv10 = nn.Conv2d(512, 512, (3, 3), (1, 1), 1)

        self.pool4 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.conv11 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)

        self.conv12 = nn.Conv2d(512, 512, (3, 3), (1, 1), 1)

        self.conv13 = nn.Conv2d(512, 512, (3, 3), (1, 1), 1)

        self.pool5 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.pool6 = nn.AvgPool2d((7, 7))

        if config.load_pretrain:
            self.load_pretrain(config.vgg_pretrain_path)


    def forward(self, x):
        net = []
        net1 = self.conv1(x)
        net1 = self.relu(net1)

        net2 = self.conv2(net1)
        net2 = self.relu(net2)

        net3 = self.pool1(net2)    # shape 56
        net3 = self.conv3(net3)
        net3 = self.relu(net3)

        net4 = self.conv4(net3)
        net4 = self.relu(net4)
        net5 = self.pool2(net4)   # shape 28

        net5 = self.conv5(net5)
        net5 = self.relu(net5)

        net6 = self.conv6(net5)
        net6 = self.relu(net6)

        net7 = self.conv7(net6)
        net7 = self.relu(net7)
        net8 = self.pool3(net7)   # shape 14

        net8 = self.conv8(net8)
        net8 = self.relu(net8)

        net9 = self.conv9(net8)
        net9 = self.relu(net9)

        net10 = self.conv10(net9)
        net10 = self.relu(net10)
        net11 = self.pool4(net10)  # shape 7

        net11 = self.conv11(net11)
        net11 = self.relu(net11)

        net12 = self.conv12(net11)
        net12 = self.relu(net12)

        net13 = self.conv13(net12)
        net13 = self.relu(net13)
        net14 = self.pool5(net13)
        return net14

    def load_pretrain(self, pretrain_path):
        check_point = torch.load(pretrain_path)
        self.load_state_dict(check_point)

    def vggLoss(self, input, target):
        scale = 224
        self.A.resize_(scale, scale).copy_(input)
        self.B.resize_(scale, scale).copy_(target)
        A = self.forward(self.A)
        B = self.forward(self.B)
        return torch.nn.L1loss(A,B)

