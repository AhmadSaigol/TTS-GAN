import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.masking import TriangularCausalMask, ProbMask
from informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from informer.attn import FullAttention, ProbAttention, AttentionLayer

class InformerEncoder(nn.Module):
    def __init__(self, 
                factor=5, 
                d_model=512, 
                n_heads=8, 
                e_layers=3, 
                d_ff=512, 
                dropout=0.0, 
                attn='prob', 
                activation='gelu', 
                output_attention = False, 
                distil=True, mix=True,
                device=torch.device('cuda:0')):
        
        super(InformerEncoder, self).__init__()
        
        self.output_attention = output_attention

        
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
       
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        
    def forward(self, x, enc_self_mask=None):
        enc_out, attns = self.encoder(x, attn_mask=enc_self_mask)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out
    

class InformerStackEncoder(nn.Module):
    def __init__(self, 
                factor=5, 
                d_model=512, 
                n_heads=8, 
                e_layers=[3,2,1],  
                d_ff=512, 
                dropout=0.0, 
                attn='prob', 
                activation='gelu',
                output_attention = False, 
                distil=True, mix=True,
                device=torch.device('cuda:0')):
        
        super(InformerStackEncoder, self).__init__()
        
        self.output_attention = output_attention

        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        
        
    def forward(self, x, enc_self_mask=None):
        
        enc_out, attns = self.encoder(x, attn_mask=enc_self_mask)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out
        

class DeconvLayer(nn.Module):
    def __init__(self, embed_dim, channels, seq_len, add_seq_block=False, red_seq_len=None, dropout=0.1):
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

if __name__ == '__main__':
    
    inf = InformerEncoder (7, 7, 14, 96, 24, 48).float()
    
    #print(inf)
    
    #x = torch.randn(32,96,512)
    #y = inf(x)
    
    #print(y.shape)
    #y = y.reshape(y.shape[0], y.shape[1], 1, y.shape[2])
    #print(y.shape)
    #y=y.permute(0, 3, 1, 2)
    #print(y.shape)
    
    #deconv = nn.Sequential(
    #    nn.Conv2d(24, 96, 1, 1, 0) # find way of detering input channels
    ##    nn.Conv2d(512, 7, 1, 1, 0)
    #)
    
    #y = deconv(y)
    #print(y.shape)
    
    #deconv = DeconvLayer(10, 3, 150, add_seq_block=True,red_seq_len= 24)
    #x = torch.randn(16,24,10)
    #y =deconv(x)
    #print(y.shape)
    
    deconv = DeconvLayer(10, 3, 150)
    
    x = torch.randn(16,150,10)
    y =deconv(x)
    print(y.shape)
    
    '''
    inf = Gen_InformerStackEncoder (7, 7, 14, 96, 24, 48)
    
    print(inf)
    
    x = torch.randn(32,96,512)
    y = inf(x)
    
    print(y.shape)
    '''
    """
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
    
    model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
    if self.args.model=='informer' or self.args.model=='informerstack':
        e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            e_layers, # self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()"""