import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# Define Residual Block
class ResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define ResNet9
class ResNet(nn.Layer):
    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, stride=1)
        self.layer2 = self._make_layer(block, 128, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        if num_classes > 0:
            self.fc = nn.Linear(128, num_classes)
        else:
            self.fc = None

    def _make_layer(self, block, out_channels, stride):
        layer = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = paddle.flatten(out, 1)
        if self.fc:
            out = self.fc(out)
        return out

# Instantiate ResNet9 model
def ResNet9(num_classes=100):
    return ResNet(ResidualBlock, num_classes)

# Create model
model = ResNet9()

# Simulate forward pass with random input tensor
input_tensor = paddle.rand([4, 3, 512, 512])  # Batch size 4, 3 channels, 32x32 image
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
