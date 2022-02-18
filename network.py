import torch
from torch import nn 
from utils import get_sinusoid_encoding_table
from module import *


class Encoder(nn.Module):
    def __init__(self, train_config, symbol_length):
        """
        Transformer Encoder

        """
        super(Encoder, self).__init__()
        
        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, 
                                                                                train_config.hidden_size, 
                                                                                padding_idx=0),
                                                                                freeze=True)
        self.pos_dropout = nn.Dropout(p=train_config.dropout_p)
        
        self.encoder_prenet = EncoderPrenet(train_config.embedding_size, 
                                            train_config.hidden_size, 
                                            symbol_length)
        
        self.blocks = clones(EncoderBlock(train_config.hidden_size, 
                                          train_config.n_head, 
                                          train_config.dropout_p), train_config.n_layers)
    
    def forward(self, x, pos):
        
        c_mask, mask = self.generate_mask(x, pos)
        x = self.encoder_prenet(x)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        return x, c_mask
    
    @torch.no_grad()
    def generate_mask(self, x, pos):
        
        c_mask = pos.ne(0).type(torch.float)
        mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        
        return c_mask, mask


class Decoder(nn.Module):
    def __init__(self, train_config, n_mels):
        """
        Transformer Decoder
        
        """
        super(Encoder, self).__init__()


        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024,
                                                                        train_config.hidden_size, padding_idx=0),
                                            freeze=True)
        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_dropout = nn.Dropout(p=0.1)
        self.decoder_prenet = DecoderPrenet(n_mels, train_config.hidden_size * 2,
                               train_config.hidden_size, dropout_p=0.2)
        
        self.norm = Linear(train_config.hidden_size, 
                           train_config.hidden_size)
        

        self.dec_blocks = clones(DecoderBlock(train_config.hidden_size,
                                          train_config.n_head,
                                          train_config.dropout_p), train_config.n_layers)
        
        
        self.mel_linear = Linear(train_config.hidden_size, 
                                 n_mels * train_config.outputs_per_step)
        self.stop_linear = Linear(
            train_config.hidden_size, 1, w_init='sigmoid')
        
        self.postconvnet = PostConvNet(train_config.hidden_size)

    def forward(self, x, encoder_output, pos, c_mask, prev=None):

        m_mask, enc_dec_mask = self.generate_mask(x, pos, c_mask)
        x = self.decoder_prenet(x)
        x = self.norm(x)
        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x
        x = self.pos_dropout(x)
        # Positional dropout

        for block in self.dec_blocks:
            x = block(x, encoder_output, 
                      enc_dec_mask, m_mask, prev)
        
        # Linear Project
        mel_out = self.mel_linear(x)
        
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        postnet_out = out.transpose(1, 2)
        
        # Stop token Prediction
        stop_tokens = self.stop_linear(x)
        
        return mel_out, postnet_out, stop_tokens

    @torch.no_grad()
    def generate_mask(self, x, pos_mel, c_mask):
        batch_size, decoder_len = x.size(0), x.size(1)
        m_mask = pos_mel.ne(0).type(torch.float)
        mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
        
        if self.training:
            m_mask = mask + torch.triu(torch.ones(decoder_len, 
                                                  decoder_len),
                                       diagonal=1).repeat(batch_size, 1, 1).byte()
            
            m_mask = m_mask.gt(0)
        else:
            m_mask = None
            
        enc_dec_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
        enc_dec_mask = enc_dec_mask.transpose(1, 2)
        return m_mask, enc_dec_mask

class TransformerTTS(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, train_config, data_config):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(train_config, data_config.symbol_length)
        self.decoder = Decoder(train_config, data_config.n_mels)

    def forward(self, texts, mel_inputs, pos_texts, pos_mels, prev=None):
        encoder_output, c_mask = self.encoder(texts, pos=pos_texts)
        mel_out, postnet_out, stop_tokens = self.decoder(mel_inputs, encoder_output, pos_mels, c_mask, prev)

        return mel_out, postnet_out, stop_tokens