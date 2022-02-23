import torch
import torch.nn as nn
from tqdm import tqdm


class Engine():
    def __init__(self, train_config, train_loader, val_loader, device):
        self.train_config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device        
        
    def loss_fn(self, true_mel, 
                true_stop_token, 
                pred_mel_post, 
                pred_mel, 
                pred_stop_token):
        
        mel_loss = nn.L1Loss()(pred_mel, true_mel) + nn.L1Loss()(pred_mel_post, true_mel)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.train_config.bce_weight))(pred_stop_token.squeeze(), true_stop_token)
        
        
        return mel_loss, bce_loss

    def train_step(self, model, optimizer, scheduler):
        running_loss = 0
        mel_losses = 0
        bce_losses = 0
        model.train()
        optimizer.zero_grad()
        for data in tqdm(self.train_loader):
            character = data['text'].to(self.device)
            mel = data['mel'].to(self.device)
            mel_input = data['mel_input'].to(self.device)
            pos_text = data['pos_text'].to(self.device)
            pos_mel = data['pos_mel'].to(self.device)
            stop_token = data['stop_tokens'].to(self.device)
            
            mel_out, postnet_out, stop_pred, enc_attn_list, mask_attn_list, enc_dec_attn_list = model(
                    character, mel_input, pos_text, pos_mel)
            mel_loss, bce_loss = self.loss_fn(mel, stop_token, postnet_out, mel_out, stop_pred)
            
            mel_losses += mel_loss
            bce_losses += bce_loss
            loss = mel_loss + bce_loss
            running_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            
        epoch_loss = running_loss/len(self.train_loader)
        mel_losses = mel_losses/len(self.train_loader)
        bce_losses = bce_losses/len(self.train_loader)
        return epoch_loss, mel_losses, bce_losses, enc_attn_list, mask_attn_list, enc_dec_attn_list
    
    
    def val_step(self, model):
        model.eval()
        running_loss = 0
        mel_losses = 0
        bce_losses = 0
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                character = data['text'].to(self.device)
                mel = data['mel'].to(self.device)
                mel_input = data['mel_input'].to(self.device)
                pos_text = data['pos_text'].to(self.device)
                pos_mel = data['pos_mel'].to(self.device)
                stop_token = data['stop_tokens'].to(self.device)

                mel_out, postnet_out, stop_pred, enc_attn_list, mask_attn_list, enc_dec_attn_list = model(
                    character, mel_input, pos_text, pos_mel)
                mel_loss, bce_loss = self.loss_fn(
                    mel, stop_token, postnet_out, mel_out, stop_pred)

                mel_losses += mel_loss
                bce_losses += bce_loss
                loss = mel_loss + bce_loss
                running_loss += loss
                    
        epoch_loss = running_loss/len(self.val_loader)
        mel_losses = mel_losses/len(self.val_loader)
        bce_losses = bce_losses/len(self.val_loader)
        return epoch_loss, mel_losses, bce_losses, enc_attn_list, mask_attn_list, enc_dec_attn_list
