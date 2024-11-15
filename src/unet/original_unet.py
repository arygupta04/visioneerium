import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

#YOU CAN CROP ASWELL - original paper has cropped it
def pad_img(tensor, target_tensor):
    # Determine the size differences
    target_height, target_width = target_tensor.size()[2], target_tensor.size()[3]
    tensor_height, tensor_width = tensor.size()[2], tensor.size()[3]

    diff_height = tensor_height - target_height
    diff_width = tensor_width - target_width

    # Crop the tensor if it is larger than the target
    if diff_height > 0 or diff_width > 0:
        tensor = tensor[:, :, diff_height // 2 : tensor_height - diff_height // 2, 
                        diff_width // 2 : tensor_width - diff_width // 2]
    return tensor

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv_block(1, 64)
        self.down_conv_2 = double_conv_block(64, 128)
        self.down_conv_3 = double_conv_block(128, 256)
        self.down_conv_4 = double_conv_block(256, 512)
        self.down_conv_5 = double_conv_block(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        self.up_conv_1 = double_conv_block(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        self.up_conv_2 = double_conv_block(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.up_conv_3 = double_conv_block(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.up_conv_4 = double_conv_block(128, 64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )
    def forward(self, image):
        # ecoder
        ec1 = self.down_conv_1(image) #
        em1 = self.max_pool_2x2(ec1)
        ec2 = self.down_conv_2(em1) #
        em2 = self.max_pool_2x2(ec2)
        ec3 = self.down_conv_3(em2) #
        em3 = self.max_pool_2x2(ec3)
        ec4 = self.down_conv_4(em3) #
        em4 = self.max_pool_2x2(ec4)
        ec5 = self.down_conv_5(em4)
        #print(dc5.size())

        #decoder
        d = self.up_trans_1(ec5)
        dc = pad_img(ec4, d)
        d = self.up_conv_1(torch.cat([d, dc], 1))

        d = self.up_trans_2(d)
        dc = pad_img(ec3, d)
        d = self.up_conv_2(torch.cat([d, dc], 1))

        d = self.up_trans_3(d)
        dc = pad_img(ec2, d)
        d = self.up_conv_3(torch.cat([d, dc], 1))

        d = self.up_trans_4(d)
        dc = pad_img(ec1, d)
        d = self.up_conv_4(torch.cat([d, dc], 1))

        d = self.out(d)
        print(d.size())
        return d
#debug testing code
if __name__ == "__main__":
    image = torch.rand((6, 3, 256, 256))  # before: (1, 1, 572, 572)
    model = UNet()
    print(model(image))

    