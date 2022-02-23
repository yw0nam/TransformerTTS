from utils import preprocess
from configs import DataConfig, TrainConfig
from dataset import *
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch import nn
import torch
from tensorboardX import SummaryWriter
from network import TransformerTTS
import torchvision.utils as vutils
import transformers
from engine import Engine

def load_loader(train_config, data_config):
    preprocessor = preprocess(data_config)
    train_dataset = LJDatasets(data_config, preprocessor, train=True)
    val_dataset = LJDatasets(data_config, preprocessor, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True,
                                  collate_fn=Transformer_Collator(preprocessor))
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False,
                                  collate_fn=Transformer_Collator(preprocessor))
    
    return train_loader, val_loader

def main(data_config, train_config):

    train_loader, val_loader = load_loader(train_config, data_config)
    
    model = nn.DataParallel(TransformerTTS(train_config, data_config).cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    num_training_steps = train_config.epochs * len(train_loader)//train_config.batch_size
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        # num_warmup_steps=int(train_config.warmup_ratio*num_training_steps),
        num_warmup_steps=train_config.warmup_step,
        num_training_steps=num_training_steps
    )
    myengine = Engine(train_config, train_loader, val_loader, "cuda")
    writer = SummaryWriter(log_dir=train_config.log_dir)
    epoch_start = 0
    
    if os.path.exists(train_config.load_checkpoint):
        checkpoint = torch.load(train_config.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f'--------- Restarting Training from Epoch {epoch_start} -----------\n')
    
    else:
        best_loss = 1e10 
    
    print('--------- STARTING TRAINING ---------\n')
    for epoch in tqdm(range(epoch_start, train_config.epochs)):
        train_epoch_loss, train_mel_losses, train_bce_losses, _, _, _ = myengine.train_step(model, optimizer, scheduler)
        val_epoch_loss, val_mel_losses, val_bce_losses, _, _ ,_  = myengine.val_step(model)

        writer.add_scalars('training_loss', {
            'Loss': train_epoch_loss,
            'mel_loss + post_mel_loss': train_mel_losses,
            'stop_loss': train_bce_losses
        }, epoch)
        writer.add_scalars('validation_loss', {
            "Loss": val_epoch_loss,
            'mel_loss + post_mel_loss': val_mel_losses,
            'stop_loss': val_bce_losses
        }, epoch)
        
        writer.add_scalars('alphas', {
            'encoder_alpha': model.module.Text_Encoder.alpha.data,
            'decoder_alpha': model.module.Mel_Decoder.alpha.data,
        }, epoch)
        
        if epoch % train_config.save_step == 0:
            try:
                torch.save({
                    'epoch'                 : epoch,
                    'model_state_dict'      : model.state_dict(),
                    'optimizer_state_dict'  : optimizer.state_dict(),
                    'scheduler_state_dict'  : scheduler.state_dict(),
                    'loss': val_epoch_loss,
                }, os.path.join(train_config.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % epoch))
            except:
                os.mkdir(train_config.checkpoint_path)
                torch.save({
                    'epoch'                 : epoch,
                    'model_state_dict'      : model.state_dict(),
                    'optimizer_state_dict'  : optimizer.state_dict(),
                    'scheduler_state_dict'  : scheduler.state_dict(),
                    'loss': val_epoch_loss,
                }, os.path.join(train_config.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % epoch))

        if best_loss > val_epoch_loss:
            best_loss = val_epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, train_config.model_save_path)
            
if __name__ == '__main__':
    data_config = DataConfig(
        root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
        train_csv='metadata_train.csv',
        val_csv='metadata_val.csv'
    )
    train_config = TrainConfig()
    main(data_config, train_config)
