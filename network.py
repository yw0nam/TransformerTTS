import torch
from torch import nn 
from utils import get_sinusoid_encoding_table
from module import *

class Encoder(nn.Module):
    def __init__(self, train_config, symbol_length):
        """
        Transformer Encoder

        Args:
            embedding_size (int): Number of embedding size, default 512
            hidden_size (int): Number of hidden size, default 256
            n_layer (int): Number of Encoder Block, default 6
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, train_config.hidden_size, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=train_config.dropout_p)
        self.encoder_prenet = EncoderPrenet(train_config.embedding_size, train_config.hidden_size, symbol_length)
        self.blocks = clones(EncoderBlock(train_config.hidden_size, train_config.n_head, train_config.dropout_p), train_config.n_layers)
    
    def forward(self, x, pos):
        
        if self.training:
            c_mask = pos.ne(0).type(torch.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None
            
        x = self.encoder_prenet(x)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x, mask = block(x, mask=mask)
            
        return x, c_mask
    