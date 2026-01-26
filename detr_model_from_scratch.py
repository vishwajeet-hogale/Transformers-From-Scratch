import torch 
import torch.nn as nn
import math
class PositionEmbedding2D(nn.Module):
    """
    2D sine-cosine positional encoding as used in DETR.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        returns: positional encoding of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        # coordinate grids
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        if self.normalize:
            y_embed = y_embed / (H - 1) * self.scale
            x_embed = x_embed / (W - 1) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(),
                             pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(),
                             pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)   # (H, W, C)
        pos = pos.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
        return pos.repeat(B, 1, 1, 1)

class ImageEncoder(nn.Module):
  def __init__(self, input_channels=3, output_channels=64, kernel_size=8, stride=8):
    super(ImageEncoder, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=1) # Can be replaced with Resnet

  def forward(self, x):
    x = self.conv1(x)
    
    return x

class IntermediateTransformerEncoder(nn.Module):
  def __init__(self, hidden_dim, num_heads, num_layers):
    super(IntermediateTransformerEncoder, self).__init__()
    self.encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead=num_heads, batch_first=True)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

  def forward(self, x):
    x = self.encoder(x)
    return x

class IntermediateTransformerDecoder(nn.Module):
  def __init__(self, hidden_dim, num_heads, num_layers, num_queries=50):
    super(IntermediateTransformerDecoder, self).__init__()
    self.queries = nn.Embedding(num_queries, hidden_dim)
    self.decoder_layer = nn.TransformerDecoderLayer(d_model = hidden_dim, nhead=num_heads, batch_first=True)
    self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

  def forward(self, x):
    B = x.shape[0]
    queries = self.queries.weight.unsqueeze(0).repeat(B, 1, 1) # Expand vs repeat : Use expand when it is in-place
    x = self.decoder(tgt= queries, memory=x)
    return x

class ClassifierHead(nn.Module):
  def __init__(self, hidden_dim, num_classes):
    super(ClassifierHead, self).__init__()
    self.head = nn.Linear(hidden_dim, num_classes+1)

  def forward(self, x):
    x = self.head(x)
    return x

class BoundingBox(nn.Module):
  def __init__(self, hidden_dim):
    super(BoundingBox, self).__init__()
    self.head = nn.Linear(hidden_dim, 4)

  def forward(self, x):
    x = self.head(x).sigmoid()
    return x

class DeTR(nn.Module):
  def __init__(self, input_channels, output_channels, hidden_dim, num_heads, num_layers, num_classes):
    super(DeTR, self).__init__()
    self.image_encoder = ImageEncoder(input_channels=input_channels, output_channels=output_channels)
    self.positional_encoding = PositionEmbedding2D(num_pos_feats=hidden_dim//2)
    self.transformer_encoder = IntermediateTransformerEncoder(hidden_dim, num_heads, num_layers)
    self.transformer_decoder = IntermediateTransformerDecoder(hidden_dim, num_heads, num_layers)
    self.classifier_head = ClassifierHead(hidden_dim, num_classes)
    self.bbox_head = BoundingBox(hidden_dim)
  def forward(self, x):
    x = self.image_encoder(x)
    pos = self.positional_encoding(x)
    x = x + pos
    batch, patches = x.shape[0], x.shape[1]
    x = x.reshape(batch, patches, -1)
    x = x.permute(0, 2, 1) # transpose is possible too

    x = self.transformer_encoder(x)
    x = self.transformer_decoder(x)

    classifier_head = self.classifier_head(x)
    bbox_head = self.bbox_head(x)
    return classifier_head, bbox_head


if __name__ == "__main__":
    batch, H, W, C = 8, 32, 32, 3
    A = torch.rand(batch, C, H, W)
    
    classifier_head, bbox_head = DeTR(input_channels=3, output_channels=64, hidden_dim=64, num_heads=4, num_layers=2, num_classes=10)(A)
    print(classifier_head.shape, bbox_head.shape)