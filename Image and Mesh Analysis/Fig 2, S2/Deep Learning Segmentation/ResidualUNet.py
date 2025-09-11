import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function: Concatenate with padding for spatial alignment and channel matching
def concat_with_padding(x, skip):
    if skip is not None:
        if x.shape[2:] != skip.shape[2:]:
            diff_depth = x.shape[2] - skip.shape[2]
            diff_height = x.shape[3] - skip.shape[3]
            diff_width = x.shape[4] - skip.shape[4]
            skip = F.pad(skip,
                         (diff_width // 2, diff_width - diff_width // 2,
                          diff_height // 2, diff_height - diff_height // 2,
                          diff_depth // 2, diff_depth - diff_depth // 2))
        x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
    return x

# Custom 3D Median Pooling function
def median_pool3d(x, kernel_size, stride):
    pad = [(k - 1) // 2 for k in kernel_size]
    x = F.pad(x, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]), mode="reflect")
    x_unf = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]).unfold(4, kernel_size[2], stride[2])
    x_unf = x_unf.contiguous().view(*x_unf.size()[:5], -1)
    median_vals = x_unf.median(dim=-1)[0]
    return median_vals

# Convolution block with configurable kernel sizes and zero padding
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, use_bn=True, residual=True):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        
        # First layer
        kernel_size = kernel_sizes[0]
        padding = [k // 2 for k in kernel_size]
        self.layers.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
            nn.ELU()
        ))

        # Residual connection after the first convolution layer
        self.identity_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Remaining layers
        for kernel_size in kernel_sizes[1:]:
            padding = [k // 2 for k in kernel_size]
            self.layers.append(nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm3d(out_channels) if use_bn else nn.Identity(),
                nn.ELU()
            ))

    def forward(self, x):
        identity = self.identity_conv(x) if self.residual else None
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        if self.residual and identity is not None:
            out += identity
        return F.elu(out) if self.residual else out

# Upsample block with transpose convolution and concatenation-based skip connection
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernels, upsample_kernel=(2, 2, 2)):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=upsample_kernel, stride=upsample_kernel)
        self.conv = ConvBlock(out_channels * 2, out_channels, kernel_sizes=conv_kernels, residual=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Concatenation with skip connection
        x = concat_with_padding(x, skip)
        return self.conv(x)

# Residual 3D U-Net with corresponding encoder-decoder paths
class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[32, 64, 128, 256, 512], pooling_methods=None):
        super().__init__()
        self.pooling_methods = pooling_methods or ["median", "avg", "max", "avg"]

        # Define encoder blocks with specified filters and unique kernel sizes for each block
        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, filters[0], kernel_sizes=[(1, 7, 7), (3, 7, 7), (3, 7, 7)]),
            ConvBlock(filters[0], filters[1], kernel_sizes=[(1, 5, 5), (5, 5, 5), (3, 5, 5)]),
            ConvBlock(filters[1], filters[2], kernel_sizes=[(1, 3, 3), (3, 3, 3), (3, 3, 3)]),
            ConvBlock(filters[2], filters[3], kernel_sizes=[(1, 3, 3), (3, 3, 3), (3, 3, 3)])
        ])
        self.pooling_kernels = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (1, 2, 2)]
        
        # Bottleneck layer
        self.bottleneck = ConvBlock(filters[3], filters[4], kernel_sizes=[(3, 3, 3), (3, 3, 3), (3, 3, 3)])

        # Define decoder blocks with upsampling layers and concatenation-based skip connections
        self.upsamples = nn.ModuleList([
            UpsampleBlock(filters[4], filters[3], conv_kernels=[(1, 3, 3), (3, 3, 3), (3, 3, 3)], upsample_kernel=(1, 2, 2)),
            UpsampleBlock(filters[3], filters[2], conv_kernels=[(1, 3, 3), (3, 3, 3), (3, 3, 3)], upsample_kernel=(2, 2, 2)),
            UpsampleBlock(filters[2], filters[1], conv_kernels=[(1, 5, 5), (5, 5, 5), (3, 5, 5)], upsample_kernel=(1, 2, 2)),
            UpsampleBlock(filters[1], filters[0], conv_kernels=[(1, 7, 7), (3, 7, 7), (3, 7, 7)], upsample_kernel=(1, 2, 2))
        ])

        # Output layer
        self.out_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

        # Apply Kaiming initialization to all layers
        self.apply(self.kaiming_init)

    def forward(self, x):
        skips = []

        # Encoder path with customizable pooling
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            # Apply pooling based on the specified pooling method
            if self.pooling_methods[i] == "max":
                x = F.max_pool3d(x, kernel_size=self.pooling_kernels[i], stride=self.pooling_kernels[i])
            elif self.pooling_methods[i] == "avg":
                x = F.avg_pool3d(x, kernel_size=self.pooling_kernels[i], stride=self.pooling_kernels[i])
            elif self.pooling_methods[i] == "median":
                x = median_pool3d(x, kernel_size=self.pooling_kernels[i], stride=self.pooling_kernels[i])

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with upsampling and concatenation-based skip connections
        for i, upsample in enumerate(self.upsamples):
            x = upsample(x, skips[-(i + 1)])  # <--- Concatenation applied here

        # Final output layer
        return torch.sigmoid(self.out_conv(x))

    # Kaiming initialization function
    @staticmethod
    def kaiming_init(layer):
        if isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
