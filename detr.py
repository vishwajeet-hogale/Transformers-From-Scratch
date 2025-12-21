import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights

class SyntheticDetection(Dataset):
    """
    Generates images with random filled rectangles.
    DETR-ready: boxes are normalized cxcywh in [0,1].
    """
    def __init__(self, n=2000, image_size=128, max_objects=3):
        self.n = n
        self.S = image_size
        self.max_objects = max_objects

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        S = self.S
        img = torch.zeros(3, S, S, dtype=torch.float32)

        num_objs = torch.randint(1, self.max_objects + 1, (1,)).item()
        boxes = []
        labels = []

        for _ in range(num_objs):
            # random rectangle size (in pixels)
            w = torch.randint(S // 10, S // 3, (1,)).item()
            h = torch.randint(S // 10, S // 3, (1,)).item()
            x0 = torch.randint(0, S - w, (1,)).item()
            y0 = torch.randint(0, S - h, (1,)).item()
            x1 = x0 + w
            y1 = y0 + h

            # draw rectangle (white)
            img[:, y0:y1, x0:x1] = 1.0

            # normalized cxcywh
            cx = (x0 + x1) / 2 / S
            cy = (y0 + y1) / 2 / S
            bw = (x1 - x0) / S
            bh = (y1 - y0) / S

            boxes.append([cx, cy, bw, bh])
            labels.append(0)  # only one class: "rect"

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)


# Usage:
batch_size = 8
img_size = 128
train_dataset = SyntheticDetection(n=2000, image_size=img_size, max_objects=3)
val_dataset   = SyntheticDetection(n=200, image_size=img_size, max_objects=3)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


class ImageResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # keep ONLY convolutional part
        self.resnet_feature_extractor = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
    def forward(self, images):
        image_features = self.resnet_feature_extractor(images)
        return image_features

class ImageTokens(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 2048
        self.layer_norm = nn.LayerNorm(self.num_channels)
        self.resnet_encoder = ImageResNetBackbone()
        self.pse = nn.Parameter(torch.randn(16, self.num_channels))
    def forward(self, image_features):
        norm_image_features = image_features.flatten(2)
        norm_image_features = norm_image_features.transpose(1,2)
        norm_image_features = self.layer_norm(norm_image_features)
        image_tokens = norm_image_features + self.pse
        return image_tokens
    
class DeTREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2048,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,  
            activation="relu",
            norm_first=True    
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = 4)
    def forward(self, image_tokens):
        outputs = self.encoder(image_tokens)
        masks = torch.zeros(outputs.shape[0], 16)
        return outputs, masks
    
class DeTRDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_queries = 100
        self.query_embed = nn.Embedding(self.num_queries, 2048)
        # Stack of decoder layers (self-attn on queries + cross-attn to memory)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=2048,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,   # so shapes are (B, L, C)
            activation="relu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)
        
    def forward(self, encoder_outputs, encoder_masks = None):
        tgt = torch.zeros(encoder_outputs.shape[0], self.num_queries, 2048, device=encoder_outputs.device)   # (B, N, C)
        tgt = tgt + self.query_embed.weight.unsqueeze(0)                  # (B, N, C)

        # Decoder:
        # - self-attn over tgt (queries)
        # - cross-attn from tgt to memory (image tokens)
        hs = self.decoder(
            tgt=tgt,
            memory=encoder_outputs,
            tgt_key_padding_mask=None, 
            memory_key_padding_mask=encoder_masks
        )  # (B, N, C)

        return hs

class DETRHeads(nn.Module):
    def __init__(self, d_model=2048, num_classes=1):
        super().__init__()
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, hs):
        pred_logits = self.class_embed(hs)        # (B, N, K+1)
        pred_boxes  = self.bbox_embed(hs).sigmoid()  # (B, N, 4)
        return pred_logits, pred_boxes

class DetectionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_feature_extractor = ImageResNetBackbone()
        self.image_tokens = ImageTokens()
        self.encoder_blocks = DeTREncoder()
        self.decoder_blocks = DeTRDecoder()
        self.heads = DETRHeads()
    def forward(self, images):
        image_features = self.image_feature_extractor(images)
        image_tokens = self.image_tokens(image_features)
        encoder_outputs, mask = self.encoder_blocks(image_tokens)
        decoder_outputs = self.decoder_blocks(encoder_outputs, mask)
        pred_logits, pred_boxes = self.heads(decoder_outputs)
        return pred_logits, pred_boxes
    
if __name__ == "__main__":
    images, targets = next(iter(train_loader))
    model = DetectionTransformer()
    outputs = model(images)


