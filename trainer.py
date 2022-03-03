from configs import DataConfig, TrainConfig
import os
from TransFormerTTS.pl_model import PL_model
import pytorch_lightning as pl
from dataset import *
import pytorch_lightning.callbacks as plc
import torch
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    pl.seed_everything(42)
    gpus = torch.cuda.device_count()
    
    out_path = './experiments/using_phonemes'
    data_config = DataConfig(
        root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
        train_csv='metadata_train.csv',
        val_csv='metadata_val.csv'
    )
    train_config = TrainConfig(
        batch_size=32,
        checkpoint_path= os.path.join(out_path, 'checkpoint'),
        log_dir= os.path.join(out_path, 'tensorboard')
    )
    model = PL_model(train_config, data_config, gpus)
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=5,
        mode="min",
    )
    
    logger = TensorBoardLogger(train_config.log_dir, name="TransformerTTS")
    data = PartitionPerEpochDataModule(train_config.batch_size, data_config, num_workers=16)

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator="ddp",
        max_epochs=train_config.epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=16, 
        amp_backend="native",
        profiler="simple",
        gradient_clip_val=1,
        logger=logger
        # auto_scale_batch_size=True
    )

    # %%
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()