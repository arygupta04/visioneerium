import torch
import torch.nn as nn
import torchvision.models as models

def double_conv_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

class UNetWithResNet50Encoder(nn.Module):
    def __init__(self, num_classes):
        super(UNetWithResNet50Encoder, self).__init__()

        # Use a pre-trained ResNet50 model as the encoder
        resnet = models.resnet50(pretrained=True)
        
        # Excluding the final FC and pooling layers
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder layers
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=2048,  # From ResNet50's last block
            out_channels=1024,
            kernel_size=2,
            stride=2
        )
        self.up_conv_1 = double_conv_block(2048, 1024)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_conv_2 = double_conv_block(1024, 512)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv_3 = double_conv_block(512, 256)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv_4 = double_conv_block(256, 128)
        
        self.out = nn.Conv2d(
            in_channels=128,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, image):
        # Encoding using ResNet50
        ec1 = self.encoder[0:3](image)  # Conv1 layer (64 channels)
        ec2 = self.encoder[3:5](ec1)    # Layer1 (256 channels)
        ec3 = self.encoder[5:6](ec2)    # Layer2 (512 channels)
        ec4 = self.encoder[6:7](ec3)    # Layer3 (1024 channels)
        ec5 = self.encoder[7:8](ec4)    # Layer4 (2048 channels)

        # Decoder using transposed convolutions
        d = self.up_trans_1(ec5)  # Upsample from 2048 channels
        d = self.up_conv_1(torch.cat([d, ec4], 1))  # Concatenate with 1024 channels

        d = self.up_trans_2(d)  # Upsample from 1024 channels
        d = self.up_conv_2(torch.cat([d, ec3], 1))  # Concatenate with 512 channels

        d = self.up_trans_3(d)  # Upsample from 512 channels
        d = self.up_conv_3(torch.cat([d, ec2], 1))  # Concatenate with 256 channels

        # To match the number of channels before concatenating with ec1, 
        # let's apply a convolutional layer to `d` to bring its channels to 256
        d = self.up_trans_4(d)  # Upsample from 256 channels
        d = self.up_conv_4(torch.cat([d, ec1], 1))  # Concatenate with 64 channels

        # Final output layer
        d = self.out(d)
        print(d.size())  # Check the output size
        return d



# Debug testing code
if __name__ == "__main__":
    image = torch.rand((6, 3, 256, 256))  # Input tensor with shape (batch_size, channels, height, width)
    model = UNetWithResNet50Encoder(num_classes=4)  # Output segmentation classes
    output = model(image)
    print(output.size())  # Expected output size: (batch_size, num_classes, height, width)


    