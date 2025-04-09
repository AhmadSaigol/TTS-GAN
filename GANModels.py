import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from informer.model import InformerEncoder, InformerStackEncoder


class Generator(nn.Module):
    def __init__(self, 
                seq_len=150, 
                patch_size=15, 
                channels=3, 
                num_classes=9, 
                latent_dim=100, 
                embed_dim=10, 
                depth=3,
                num_heads=5, 
                forward_drop_rate=0.5, 
                attn_drop_rate=0.5,
                
                model_type = 'transformer',
                 
                factor=5, 
                attn='prob', 
                ):
        super(Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        # NOTE: ADDED BY AHMAD
        self.num_heads = num_heads
        self.model_type = model_type
        
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        
        # max pool layer decreases seqlen during encoder layers.
        # to find length after encoder, we can use the following function
        # padding = 1, kernel = 3, stride = 2 
        len_after_encoder = lambda x: int(np.floor((x + 2*1-(3-1)-1)/2 + 1))
        
        if model_type == 'transformer':
            self.blocks = Gen_TransformerEncoder(
                            depth=self.depth,
                            emb_size = self.embed_dim,
                            drop_p = self.attn_drop_rate,
                            forward_drop_p=self.forward_drop_rate,
                            num_heads =self.num_heads#NOTE:ADDED BY AHMAD
                            )
            add_seq_block = False
            red_seq_len = 0
        
        elif model_type == 'informer':
            self.blocks = InformerEncoder(
                factor=factor, 
                d_model=self.embed_dim, 
                n_heads=self.num_heads, 
                e_layers=self.depth,
                d_ff=None, 
                dropout=self.attn_drop_rate,
                attn=attn, 
                activation='gelu'
            )
            
            add_seq_block = True
            #red_seq_len = seq_len//(2**(self.depth-1))
            red_seq_len = seq_len
            for _ in range(self.depth-1):
                red_seq_len = len_after_encoder(red_seq_len)
            
            
        elif model_type == 'informerstack':
            self.blocks = InformerStackEncoder(
                factor=factor, 
                d_model=self.embed_dim, 
                n_heads=self.num_heads, 
                e_layers=self.depth,
                d_ff=None, 
                dropout=self.attn_drop_rate,
                attn=attn, 
                activation='gelu'
            )
            
            add_seq_block = True
            red_seq_len = seq_len
            for _ in range(max(self.depth)-1):
                red_seq_len = len_after_encoder(red_seq_len)
            
            red_seq_len *= len(self.depth)
        
        else:
            raise ValueError(f"Unknown type of model type encountered: {model_type}")
        
        self.deconv = DeconvLayer(self.embed_dim,
                                  self.channels, 
                                  self.seq_len, 
                                  add_seq_block=add_seq_block,
                                  red_seq_len=red_seq_len,
                                  dropout=self.attn_drop_rate)
        
        #self.deconv = nn.Sequential(
        #    nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        #)

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        
        #x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        #output = self.deconv(x.permute(0, 3, 1, 2))
        #output = output.view(-1, self.channels, H, W)
        # ADDED BY AHMAD
        output = self.deconv(x)
        
        return output
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x        
        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=3,
                 patch_size=15,
                 emb_size=50, 
                 seq_length = 150,
                 depth=3, 
                 n_classes=1,
                 num_heads=5,
                 model_type = 'transformer', 
                 
                 drop_p=0.5,
                 forward_drop_p=0.5,
                 factor=5,
                 attn='prob',
                 ):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            
            (
                Dis_TransformerEncoder(depth, emb_size=emb_size, 
                                       num_heads=num_heads,
                                       drop_p= drop_p,
                                       forward_drop_p=forward_drop_p) if model_type == 'transformer'
                else 
                    InformerEncoder(
                        factor=factor, 
                        d_model=emb_size, 
                        n_heads=num_heads, 
                        e_layers=depth,
                        d_ff=None, 
                        dropout=drop_p,
                        attn=attn, 
                        activation='gelu') if model_type == 'informer' 
                
                else 
                    InformerStackEncoder(
                        factor=factor, 
                        d_model=emb_size, 
                        n_heads=num_heads, 
                        e_layers=depth,
                        d_ff=None, 
                        dropout=drop_p,
                        attn=attn, 
                        activation='gelu')
             
            ),
            
            ClassificationHead(emb_size, n_classes)
        )

# NOTE: ADDED BY AHMAD        
class DeconvLayer(nn.Module):
    def __init__(self, embed_dim, channels, seq_len, add_seq_block=False, red_seq_len=0, dropout=0.1):
        super(DeconvLayer, self).__init__()
        
        self.add_seq_block = add_seq_block
        self.seq_len = seq_len
        self.channels = channels
        
        self.deconv_channels = nn.Conv2d(embed_dim, channels, 1, 1, 0)
        
        if add_seq_block:
            #self.deconv_seq = nn.Conv2d(red_seq_len, seq_len, 1, 1, 0)
            self.deconv_seq = nn.LazyConv2d(seq_len, 1, 1, 0)
            self.activation = nn.ELU()
            self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        
        if self.add_seq_block:
            x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
            x = self.deconv_seq(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x.permute(0, 3, 2, 1)
        else:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            x = x.permute(0, 3, 1, 2)
        
        x = self.deconv_channels(x)
        x = x.view(-1, self.channels, 1, self.seq_len)
        return x  
