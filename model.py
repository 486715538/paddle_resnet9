import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义残差块
class BottleneckBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)

        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)

        self.conv3 = nn.Conv2D(out_channels, out_channels * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_channels * 4)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet-50
class ResNet50(nn.Layer):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        if num_classes > 0:
            self.fc = nn.Linear(512 * 4, num_classes)
        else:
            self.fc = None

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2D(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels * 4),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        return x

# 实例化ResNet-50模型
def ResNet50Model(num_classes=1000):
    return ResNet50(BottleneckBlock, [3, 4, 6, 3], num_classes)

# 定义 Transformer 模块
class TransformerLayer(nn.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head attention
        attn_out = self.attention(x, x, x)[0]
        x = self.layernorm1(x + attn_out)
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x
# 主函数
def main():
    # 创建模型
    model = ResNet50Model(num_classes=10)

    # 模拟输入张量，执行前向传播
    input_tensor = paddle.rand([4, 3, 224, 224])  # batch size 4, 3 channels, 224x224 image
    output = model(input_tensor)

    # 打印输出形状
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
