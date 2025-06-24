import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_same_padding(kernel_size, stride=1, dilation=1):
    
    """Compute symmetric padding for 1D convolutions to approximate 'same' output length.

    Formula: padding = (dilation * (kernel_size - 1) + stride - 1) / 2, rounded to nearest int.

    Args:
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride value. Default = 1.
        dilation (int, optional): Dilation value. Default = 1.

    Returns:
        int: Padding size to apply on both sides of the temporal dimension.
    """
    
    padding = (dilation * (kernel_size - 1) + stride - 1) / 2
    return int(padding + 0.5)

class TCSConv(nn.Module):
    
    """
    Time-Channel Separable 1D Convolution:
    Implements a depthwise convolution followed by a pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        stride (int, optional): Stride for the depthwise convolution. Default = 1.
        dilation (int, optional): Dilation for the depthwise convolution. Default = 1.

    Structure:
        - depthwise_conv: nn.Conv1d(in_channels, in_channels, kernel_size,
                        stride=stride, padding=calc_same_padding(...),
                        dilation=dilation, groups=in_channels, bias=False)
        - pointwise_conv: nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    Forward:
        Input: x of shape (batch_size, in_channels, T)
        Output: tensor of shape (batch_size, out_channels, T_out),
                where T_out depends on stride/dilation/padding.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TCSConv, self).__init__()
        padding = calc_same_padding(kernel_size, stride, dilation)

        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

    def forward(self, x):
        
        """
        Args:
            x (Tensor): shape (batch_size, in_channels, T)
        Returns:
            Tensor: shape (batch_size, out_channels, T_out).
        """
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SubBlock(nn.Module):
    
    """
    A sub-block in MatchboxNet:
    Applies TCSConv -> BatchNorm -> (optional residual add) -> ReLU -> Dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for TCSConv.
        dilation (int, optional): Dilation for TCSConv. Default = 1.
        dropout (float, optional): Dropout probability. Default = 0.2.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(SubBlock, self).__init__()
        self.tcs_conv = TCSConv(
            in_channels, out_channels,
            kernel_size, dilation=dilation
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        
        """
        Args:
            x (Tensor): shape (batch_size, in_channels, T)
            residual (Tensor or None): if provided, shape (batch_size, out_channels, T),
                added after batchnorm and before activation.

        Returns:
            Tensor: shape (batch_size, out_channels, T) after activation and dropout.
        """
        
        x = self.tcs_conv(x)
        x = self.bnorm(x)

        if residual is not None:
            x = x + residual

        x = self.relu(x)
        return self.dropout(x)

class MainBlock(nn.Module):
    
    """
    Residual block in MatchboxNet containing R SubBlocks, with skip connection at the last.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for each SubBlock.
        R (int, optional): Number of SubBlocks in this block. Default = 3.
        dilation (int, optional): Dilation passed to SubBlocks. Default = 1.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, R=3, dilation=1):
        super(MainBlock, self).__init__()
        
        # If channel dims differ, prepare 1x1 conv + BatchNorm for residual transformation
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None
        self.residual_bn = nn.BatchNorm1d(out_channels) if in_channels != out_channels else None

        self.sub_blocks = nn.ModuleList()
        # First sub-block: in_channels -> out_channels if needed
        self.sub_blocks.append(
            SubBlock(in_channels, out_channels, kernel_size, dilation, dropout=0.2)
        )

        # Remaining sub-blocks: out_channels -> out_channels
        for _ in range(1, R):
            self.sub_blocks.append(
                SubBlock(out_channels, out_channels, kernel_size, dilation, dropout=0.2)
            )

    def forward(self, x):
        
        """
        Args:
            x (Tensor): shape (batch_size, in_channels, T)
        Returns:
            Tensor: shape (batch_size, out_channels, T_out)

        Behavior:
            - Compute 'residual': if in_channels != out_channels, apply 1x1 conv + BN to x; else residual = x.
            - For each SubBlock i in self.sub_blocks:
                - If i is the last index (i == R-1), call sub_block(x, residual) to add skip connection.
                - Else, call sub_block(x) without skip.
        """
        
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        if self.residual_bn is not None:
            residual = self.residual_bn(residual)

        for i, sub_block in enumerate(self.sub_blocks):
    
            if i == len(self.sub_blocks) - 1:
                x = sub_block(x, residual)
            else:
                x = sub_block(x)
        return x

class MatchboxNet(nn.Module):
    
    """
    Full MatchboxNet architecture for audio classification / keyword spotting.
    Consists of:
      - Prologue: initial Conv1d + BatchNorm + ReLU (with stride=2 to reduce temporal length).
      - B MainBlocks in series, each containing R SubBlocks.
      - Epilogue: sequence of TCSConv + BN + ReLU layers, final Conv1d to num_classes, then AdaptiveAvgPool1d(1).

    Args:
        B (int, optional): Number of MainBlocks. Default = 3.
        R (int, optional): Number of SubBlocks per MainBlock. Default = 2.
        C (int, optional): Number of internal channels in MainBlocks. Default = 64.
        kernel_sizes (list[int] or None, optional): List of kernel sizes for each MainBlock.
            If None, defaults to [13,15,17,19,21]. Must have length >= B.
        num_classes (int, optional): Number of output classes. Default = 30.
        input_channels (int, optional): Number of input channels (e.g., n_mfcc). Default = 64.
    """
    
    def __init__(self, B=3, R=2, C=64, kernel_sizes=None, num_classes=30, input_channels=64):
        super(MatchboxNet, self).__init__()
        self.B = B
        self.R = R
        self.C = C

        if kernel_sizes is None:
            kernel_sizes = [13, 15, 17, 19, 21]
        # Ensure kernel_sizes length >= B when instantiating.


        # Prologue: Conv1d(input_channels->128, kernel_size=11, stride=2, padding same) + BN + ReLU
        self.prologue = nn.Sequential(
            nn.Conv1d(input_channels, 128, 11, stride=2, padding=calc_same_padding(11, stride=2)),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

       # Main blocks: B blocks
        self.blocks = nn.ModuleList()
        
        # First block: 128 -> C channels
        self.blocks.append(MainBlock(128, C, kernel_sizes[0], R))

        # Remaining blocks: C -> C channels
        for i in range(1, B):
            self.blocks.append(MainBlock(C, C, kernel_sizes[i], R))

        # Epilogue:
        #  - TCSConv(C->128, kernel_size=29, dilation=2) + BN + ReLU
        #  - TCSConv(128->128, kernel_size=1) + BN + ReLU
        #  - Conv1d(128->num_classes, kernel_size=1)
        #  - AdaptiveAvgPool1d(1) to aggregate temporal dimension
        self.epilogue = nn.Sequential(
            TCSConv(C, 128, 29, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            TCSConv(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, num_classes, 1),
            nn.AdaptiveAvgPool1d(1)
        )



    def forward(self, x):
        
        """
        Forward pass.

        Args:
            x (Tensor): shape (batch_size, input_channels, T)
                - input_channels typically equals n_mfcc.
                - T is the number of time frames (fixed_length).
        Returns:
            Tensor: logits of shape (batch_size, num_classes).

        Behavior:
            1. Pass through prologue: may reduce temporal length via stride=2.
            2. Pass through each MainBlock in self.blocks.
            3. Pass through epilogue, output shape (batch_size, num_classes, 1), then squeeze to (batch_size, num_classes).
            4. Note: no softmax here; returns raw logits for use with CrossEntropyLoss or manual softmax.
        """
        
        x = self.prologue(x)

        for block in self.blocks:
            x = block(x)

        x = self.epilogue(x)  #(batch_size, num_classes, 1)
        x = x.squeeze(2)      #(batch_size, num_classes)

        return x



from transformers import PreTrainedModel
from .configuration_matchboxnet import MatchboxNetConfig

class MatchboxNetForAudioClassification(PreTrainedModel):
    
    """
    Hugging Face wrapper for MatchboxNet, compatible with Trainer / AutoModelForAudioClassification.

    Args:
        config (MatchboxNetConfig): Contains parameters:
            - input_channels: number of input channels (n_mfcc)
            - num_classes: number of target classes
            - B, R, C: architecture hyperparameters
            - kernel_sizes: list of kernel sizes
            etc.
    """
    
    config_class = MatchboxNetConfig
    base_model_prefix = "matchboxnet"
    
    def __init__(self, config: MatchboxNetConfig):
        super().__init__(config)
        self.matchbox = MatchboxNet(
            B=config.B,
            R=config.R,
            C=config.C,
            kernel_sizes=config.kernel_sizes,
            num_classes=config.num_classes,
            input_channels=config.input_channels
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def _init_weights(self, module):
        
        """
        Weight initialization called by init_weights():
        - Conv1d: Kaiming normal
        - BatchNorm1d: weight=1, bias=0
        - Linear: Xavier uniform, bias=0
        """
        
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids=None, labels=None, **kwargs):
        
        """
        Forward compatible with Trainer.

        Args:
            input_ids (Tensor or array-like): shape (batch_size, input_channels, T).
                If not a Tensor, converted via torch.tensor(input_ids).
                Note: although named input_ids, this holds MFCC features (key often 'input_values').
            labels (Tensor, optional): shape (batch_size,), integer class labels.
            **kwargs: ignored here (e.g., attention_mask not used).
        Returns:
            dict:
                - "logits": Tensor of shape (batch_size, num_classes).
                - "loss": CrossEntropyLoss(logits, labels) if labels provided, else None.
        """
        
        logits = self.matchbox(input_ids)  # (B, num_classes)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss
        return output
