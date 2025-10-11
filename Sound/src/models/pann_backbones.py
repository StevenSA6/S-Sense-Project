"""
PANN (PANNs: Large-Scale Pretrained Audio Neural Networks) backbone models.
Based on: https://github.com/qiuqiangkong/audioset_tagging_cnn

These models are pretrained on AudioSet and can be used as feature extractors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from pathlib import Path


def init_layer(layer):
  """Initialize a Linear or Convolutional layer."""
  nn.init.xavier_uniform_(layer.weight)
  if hasattr(layer, "bias"):
    if layer.bias is not None:
      layer.bias.data.fill_(0.0)


def init_bn(bn):
  """Initialize a Batch Normalization layer."""
  bn.bias.data.fill_(0.0)
  bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    self.conv2 = nn.Conv2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.init_weight()

  def init_weight(self):
    init_layer(self.conv1)
    init_layer(self.conv2)
    init_bn(self.bn1)
    init_bn(self.bn2)

  def forward(self, input, pool_size=(2, 2), pool_type="avg"):
    x = input
    x = F.relu_(self.bn1(self.conv1(x)))
    x = F.relu_(self.bn2(self.conv2(x)))
    if pool_type == "max":
      x = F.max_pool2d(x, kernel_size=pool_size)
    elif pool_type == "avg":
      x = F.avg_pool2d(x, kernel_size=pool_size)
    elif pool_type == "avg+max":
      x1 = F.avg_pool2d(x, kernel_size=pool_size)
      x2 = F.max_pool2d(x, kernel_size=pool_size)
      x = x1 + x2
    else:
      raise Exception("Incorrect argument!")
    return x


class ConvPreWavBlock(nn.Module):
  """1D Convolutional block for wavegram preprocessing."""

  def __init__(self, in_channels, out_channels):
    super(ConvPreWavBlock, self).__init__()

    self.conv1 = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    self.conv2 = nn.Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        dilation=2,
        padding=2,
        bias=False
    )
    self.bn1 = nn.BatchNorm1d(out_channels)
    self.bn2 = nn.BatchNorm1d(out_channels)
    self.init_weight()

  def init_weight(self):
    init_layer(self.conv1)
    init_layer(self.conv2)
    init_bn(self.bn1)
    init_bn(self.bn2)

  def forward(self, input, pool_size):
    x = input
    x = F.relu_(self.bn1(self.conv1(x)))
    x = F.relu_(self.bn2(self.conv2(x)))
    x = F.max_pool1d(x, kernel_size=pool_size)
    return x


class Wavegram_Cnn14(nn.Module):
  """
  Wavegram_Cnn14: Takes raw waveform as input.
  Uses learnable wavegram layer followed by Cnn14 architecture.
  """

  def __init__(self, classes_num=527):
    super(Wavegram_Cnn14, self).__init__()

    self.pre_conv0 = nn.Conv1d(
        in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False
    )
    self.pre_bn0 = nn.BatchNorm1d(64)
    self.pre_block1 = ConvPreWavBlock(64, 64)
    self.pre_block2 = ConvPreWavBlock(64, 128)
    self.pre_block3 = ConvPreWavBlock(128, 128)
    self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

    self.bn0 = nn.BatchNorm2d(64)

    self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
    self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
    self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
    self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
    self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
    self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

    self.fc1 = nn.Linear(2048, 2048, bias=True)
    self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    self.init_weight()

  def init_weight(self):
    init_layer(self.pre_conv0)
    init_bn(self.pre_bn0)
    init_bn(self.bn0)
    init_layer(self.fc1)
    init_layer(self.fc_audioset)

  def forward(self, input, return_features=True):
    """
    Args:
        input: (batch_size, data_length) - raw waveform
        return_features: if True, return intermediate features instead of classification logits

    Returns:
        If return_features=True: (batch_size, feature_dim, freq_bins, time_frames)
        If return_features=False: dict with 'clipwise_output' and 'embedding'
    """
    # Wavegram preprocessing
    a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
    a1 = self.pre_block1(a1, pool_size=4)
    a1 = self.pre_block2(a1, pool_size=4)
    a1 = self.pre_block3(a1, pool_size=4)
    a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
    a1 = self.pre_block4(a1, pool_size=(2, 1))

    x = a1
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)

    if return_features:
      # Return convolutional features for downstream tasks
      # x shape: (batch_size, 2048, freq_bins, time_frames)
      return x

    # Original classification head
    x = torch.mean(x, dim=3)
    (x1, _) = torch.max(x, dim=2)
    x2 = torch.mean(x, dim=2)
    x = x1 + x2
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu_(self.fc1(x))
    embedding = F.dropout(x, p=0.5, training=self.training)
    clipwise_output = torch.sigmoid(self.fc_audioset(x))

    output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}
    return output_dict


class Wavegram_Logmel_Cnn14(nn.Module):
  """
  Wavegram_Logmel_Cnn14: Takes raw waveform as input.
  Fuses wavegram and log-mel features (log-mel computed internally in original).
  """

  def __init__(self, classes_num=527):
    super(Wavegram_Logmel_Cnn14, self).__init__()

    # Wavegram branch
    self.pre_conv0 = nn.Conv1d(
        in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False
    )
    self.pre_bn0 = nn.BatchNorm1d(64)
    self.pre_block1 = ConvPreWavBlock(64, 64)
    self.pre_block2 = ConvPreWavBlock(64, 128)
    self.pre_block3 = ConvPreWavBlock(128, 128)
    self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

    # Log-mel branch
    self.bn0 = nn.BatchNorm2d(64)

    # Fusion
    self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
    self.conv_block2 = ConvBlock(
        in_channels=128, out_channels=128)  # Fused: 64 + 64
    self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
    self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
    self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
    self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

    self.fc1 = nn.Linear(2048, 2048, bias=True)
    self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    self.init_weight()

  def init_weight(self):
    init_layer(self.pre_conv0)
    init_bn(self.pre_bn0)
    init_bn(self.bn0)
    init_layer(self.fc1)
    init_layer(self.fc_audioset)

  def forward(self, input, return_features=True):
    """
    Args:
        input: (batch_size, data_length) - raw waveform
        return_features: if True, return intermediate features

    Returns:
        If return_features=True: (batch_size, feature_dim, freq_bins, time_frames)
        If return_features=False: dict with 'clipwise_output' and 'embedding'

    Note: Original model computes log-mel spectrogram internally using spectrogram_extractor
    and logmel_extractor. This simplified version for feature extraction uses only the
    wavegram path. To use the full model, you'd need to add those modules.
    """
    # Wavegram branch
    a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
    a1 = self.pre_block1(a1, pool_size=4)
    a1 = self.pre_block2(a1, pool_size=4)
    a1 = self.pre_block3(a1, pool_size=4)
    a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
    a1 = self.pre_block4(a1, pool_size=(2, 1))

    # For simplified inference, create dummy log-mel branch
    # In full implementation, would compute: x = self.spectrogram_extractor(input)
    # then x = self.logmel_extractor(x), then x = self.conv_block1(x)
    # For now, use zeros with shape matching a1 after conv_block1 pooling
    # conv_block1 does pool_size=(2,2) which halves spatial dims
    # We need to match the output of conv_block1 to a1's shape
    x = self.conv_block1(torch.zeros((a1.shape[0], 1, a1.shape[2]*2, a1.shape[3]*2),
                                     device=a1.device, dtype=a1.dtype),
                         pool_size=(2, 2), pool_type='avg')

    # Concatenate Wavegram and Log mel spectrogram along the channel dimension
    x = torch.cat((x, a1), dim=1)

    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)

    if return_features:
      # Return convolutional features
      return x

    # Original classification head
    x = torch.mean(x, dim=3)
    (x1, _) = torch.max(x, dim=2)
    x2 = torch.mean(x, dim=2)
    x = x1 + x2
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu_(self.fc1(x))
    embedding = F.dropout(x, p=0.5, training=self.training)
    clipwise_output = torch.sigmoid(self.fc_audioset(x))

    output_dict = {"clipwise_output": clipwise_output, "embedding": embedding}
    return output_dict


def load_pretrained_backbone(backbone_type: str, checkpoint_path: str, freeze: bool = True):
  """
  Load a pretrained PANN backbone model.

  Args:
      backbone_type: Type of backbone ("wavegram_cnn14" or "wavegram_logmel_cnn14")
      checkpoint_path: Path to the checkpoint file or directory containing checkpoints
      freeze: Whether to freeze the backbone parameters

  Returns:
      Loaded backbone model
  """
  # Map backbone type to model class and checkpoint filename
  backbone_map = {
      "wavegram_cnn14": {
          "class": Wavegram_Cnn14,
          "filename": "Wavegram_Cnn14_mAP=0.389.pth"
      },
      "wavegram_logmel_cnn14": {
          "class": Wavegram_Logmel_Cnn14,
          "filename": "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
      }
  }

  if backbone_type not in backbone_map:
    raise ValueError(
        f"Unknown backbone type: {backbone_type}. "
        f"Available options: {list(backbone_map.keys())}"
    )

  # Create model
  model_class = backbone_map[backbone_type]["class"]
  model = model_class()

  # Load checkpoint
  ckpt_path = Path(checkpoint_path)
  if ckpt_path.is_dir():
    ckpt_path = ckpt_path / backbone_map[backbone_type]["filename"]

  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  print(f"Loading pretrained backbone from: {ckpt_path}")
  checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

  # Extract model state dict (checkpoint format: {'iteration': ..., 'model': {...}})
  if "model" in checkpoint:
    state_dict = checkpoint["model"]
  else:
    state_dict = checkpoint

  # Load weights (strict=False to allow loading without spectrogram_extractor/logmel_extractor)
  missing_keys, unexpected_keys = model.load_state_dict(
      state_dict, strict=False)

  # Filter out expected missing/unexpected keys
  expected_missing = ["spectrogram_extractor",
                      "logmel_extractor", "spec_augmenter"]
  missing_keys = [k for k in missing_keys if not any(
      em in k for em in expected_missing)]
  expected_unexpected = ["spectrogram_extractor",
                         "logmel_extractor", "spec_augmenter"]
  unexpected_keys = [k for k in unexpected_keys if not any(
      eu in k for eu in expected_unexpected)]

  if missing_keys:
    print(f"Warning: Missing keys in state_dict (first 5): {missing_keys[:5]}")
  if unexpected_keys:
    print(
        f"Warning: Unexpected keys in state_dict (first 5): {unexpected_keys[:5]}")

  print(f"Successfully loaded {backbone_type} checkpoint")

  # Freeze parameters if requested
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
    model.eval()
    print("Backbone parameters frozen (requires_grad=False)")

  return model
