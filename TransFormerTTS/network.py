import torch
from torch import nn 
from utils import get_sinusoid_encoding_table
from TransFormerTTS.module import *


class Text_Encoder(nn.Module):
    def __init__(self, train_config, symbol_length):
        """
        Transformer Encoder

        """
        super(Text_Encoder, self).__init__()
        
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
    
    def forward(self, x, pos, mask):
        
        x = self.encoder_prenet(x)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)
        
        attn_list = []
        for block in self.blocks:
            x, attn = block(x, mask=mask)
            attn_list.append(attn.detach())
        return x, attn_list
    


class Mel_Decoder(nn.Module):
    def __init__(self, train_config, n_mels):
        """
        Transformer Decoder
        
        """
        super(Mel_Decoder, self).__init__()


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
        self.stop_linear = nn.Sequential(
            Linear(train_config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.postconvnet = PostConvNet(train_config.hidden_size, n_mels, n_mels)

    def forward(self, x, pos, encoder_output, enc_dec_mask, dec_mask, dec_attn_mask):

        x = self.decoder_prenet(x)
        x = self.norm(x)
        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x
        x = self.pos_dropout(x)
        # Positional dropout

        mask_attn_list = []
        enc_dec_attn_list = []
        for block in self.dec_blocks:
            x, mask_attn, enc_dec_attn = block(x, encoder_output, 
                                                    enc_dec_mask, dec_mask, dec_attn_mask)
            mask_attn_list.append(mask_attn.detach())
            enc_dec_attn_list.append(enc_dec_attn.detach())
        # Linear Project
        mel_out = self.mel_linear(x)
        
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        postnet_out = out.transpose(1, 2)
        
        # Stop token Prediction
        stop_tokens = self.stop_linear(x)
        
        return mel_out, postnet_out, stop_tokens, mask_attn_list, enc_dec_attn_list

class TransformerTTS(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, train_config, data_config, return_attn=False):
        super(TransformerTTS, self).__init__()
        self.Text_Encoder = Text_Encoder(train_config, data_config.symbol_length)
        self.Mel_Decoder = Mel_Decoder(train_config, data_config.n_mels)
        self.return_attn = return_attn

    def forward(self, text, mel_input, pos_text, pos_mel):
        src_mask, trg_mask, triu_mask = self.generate_mask(pos_text, pos_mel, mel_input)
        
        encoder_output, enc_attn_list = self.Text_Encoder(text, pos_text, src_mask)
        mel_out, postnet_out, stop_tokens, mask_attn_list, enc_dec_attn_list = self.Mel_Decoder(mel_input, 
                                                                                                pos_mel,
                                                                                                encoder_output, 
                                                                                                src_mask, 
                                                                                                trg_mask, 
                                                                                                triu_mask)
        if self.return_attn:
            return {'pred_mel':mel_out, 
                    'pred_mel_post' :postnet_out, 
                    'pred_stop_tokens' :stop_tokens, 
                    'enc_attn_list' :enc_attn_list, 
                    'mask_attn_list': mask_attn_list, 
                    'enc_dec_attn_list': enc_dec_attn_list
                    }
        else:
            return {
                'pred_mel': mel_out,
                'pred_mel_post': postnet_out,
                'pred_stop_tokens': stop_tokens,
                }
    
    def generate_mask(self, pos_src, pos_trg, trg):
        with torch.no_grad():
            src_mask = pos_src.lt(1)
            trg_mask = pos_trg.lt(1)
            # triu_mask = torch.triu(torch.ones(trg.size(1), trg.size(1)), diagonal=1).bool().cuda()
            triu_mask = torch.triu(torch.ones(trg.size(1), trg.size(1)), diagonal=1).bool()
            triu_mask = triu_mask.type_as(pos_src).bool()
        return src_mask, trg_mask, triu_mask