import torch 
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


num_classes = 10
batch_size = 64
channels = 1
img_size = 28
patch_size = 7
num_patches = (img_size // patch_size) ** 2
lr = 1e-3
num_epochs = 10
embedding_dim = 64
attention_heads = 4
transformer_blocks = 16
mlp_head_nodes = 128
num_heads = 4
epochs = 10


transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

train_dataset = tv.datasets.MNIST(root="./data", train=True, download= True, transform=transforms)
val_dataset = tv.datasets.MNIST(root="./data", train=False, download= True, transform=transforms)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)




## Part 1 : Patch Embeddings
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(channels, embedding_dim, kernel_size = patch_size, stride = patch_size)
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1,2) # This is to get the hidden dim (64) to the end      
        return x # (B, 16, 64)


"""
Let's say the image is of the shape 3, 28, 28
Assuming each patch is 7 x 7
num_patches = 28/7, 28/7 = 4, 4 = 4 * 4 = 16 patches of each 4x4


If it is a batch, then B, 1, 28, 28 = B, 64, 4, 4 = flatten(2) = B, 64, 16. Now, let's transpose to 
put the embed_dim to the end. If I do transpose(1,2) = (B, 16, 64) 

This means that each token has an embeddim of 64. 
"""        


## Part 2 : Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads = transformer_blocks, batch_first = True) 
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_head_nodes),
            nn.GELU(),
            nn.Linear(mlp_head_nodes, embedding_dim)
        )
        
    def forward(self, x):
        layer_norm_output1 = self.layer_norm1(x)
        attn_out, _ = self.multihead_attention(layer_norm_output1, layer_norm_output1, layer_norm_output1)
        skip_conn1 = attn_out + x
        layer_norm_output2 = self.layer_norm2(skip_conn1)
        projected_tokens = self.mlp(layer_norm_output2)
        projected_tokens_with_skip = projected_tokens + skip_conn1
        return projected_tokens_with_skip
    
## Part 3 : MLP Head
class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, mlp_head_nodes),
            nn.ReLU(),
            nn.Linear(mlp_head_nodes, num_classes)
        )
    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp_head(x)

## Part 4 : Put it all together VisionTransformer
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_tokens = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1,num_patches + 1, embedding_dim))
        self.encoder = TransformerEncoder()
        self.encoder_blocks = nn.Sequential(
            *[TransformerEncoder() for _ in range(num_heads)]
        )
        self.mlp_head = MLPHead()
        
    def forward(self, images):
        B = images.shape[0]
        patch_tokens = self.patch_tokens(images)
        cls_token_for_batch = self.cls_token.expand(B, 1, embedding_dim)
        all_tokens = torch.concat([cls_token_for_batch, patch_tokens], dim =1)
        all_tokens_with_positional_embeddings = all_tokens + self.position_embedding
        attention_results = self.encoder_blocks(all_tokens_with_positional_embeddings)
        attention_cls = attention_results[:, 0]
        return self.mlp_head(attention_cls)
    
    
if __name__ == "__main__":
    device = torch.device('mps')
    model = VisionTransformer().to(device)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    
    ## Training
    model.train()
    total_loss = 0.0
    train_correct = 0
    for epoch in range(epochs):
        for (images, labels) in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            B = images.shape[0]
            optim.zero_grad()
            outputs = model(images)
            probs = torch.softmax(outputs, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            correct = (preds == labels).sum().item()
            acc = 100 * (correct/B)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            print(f"Loss : {total_loss}")
            
        
    
    
        




