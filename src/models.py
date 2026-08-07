import torch
from torch import nn

# based on the implementation in https://github.com/lyyraaa/spatial-spectral-prostate/blob/main/src/models.py
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()

        # Normalisation Layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

        # Fc Layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

        # Additional kit
        self.activation = nn.GELU()

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation,
            self.bn1,
            self.fc2,
            self.activation,
            self.bn2,
            self.fc3,
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

# based on the implementation in https://github.com/milesial/Pytorch-UNet
class UNet(nn.Module):
    def __init__(self, input_dim, reduce_dim, n_classes, reduce_method, bilinear=False): # todo mention pytorch unet
        super(UNet, self).__init__()

        self.n_channels = reduce_dim
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (self.DoubleConv(reduce_dim, 64))
        self.down1 = (self.Down(64, 128))
        self.down2 = (self.Down(128, 256))
        self.down3 = (self.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (self.Down(512, 1024 // factor))
        self.up1 = (self.Up(1024, 512 // factor, bilinear))
        self.up2 = (self.Up(512, 256 // factor, bilinear))
        self.up3 = (self.Up(256, 128 // factor, bilinear))
        self.up4 = (self.Up(128, 64, bilinear))
        self.outc = (self.OutConv(64, n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()

            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = UNet.DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = UNet.DoubleConv(in_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                UNet.DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)