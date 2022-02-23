# %%
from configs import DataConfig, TrainConfig
import os
from TransFormerTTS.pl_model import PL_model
import pytorch_lightning as pl
from dataset import *
import pytorch_lightning.callbacks as plc
import torch
from torch.utils.data import DataLoader
# %%


def load_loader(train_config, data_config):
    preprocessor = preprocess(data_config)
    train_dataset = LJDatasets(data_config, preprocessor, train=True)
    val_dataset = LJDatasets(data_config, preprocessor, train=False)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True,
                              collate_fn=Transformer_Collator(preprocessor), num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=True,
                            collate_fn=Transformer_Collator(preprocessor), num_workers=16)

    return train_loader, val_loader

def main():
    pl.seed_everything(42)
    gpus = torch.cuda.device_count()
    data_config = DataConfig(
        root_dir="/home/spow12/data/TTS/LJSpeech-1.1/",
        train_csv='metadata_train.csv',
        val_csv='metadata_val.csv'
    )
    train_config = TrainConfig()
    # %%
    model = PL_model(train_config, data_config, gpus)
    if not os.path.exists(train_config.checkpoint_path):
        os.makedirs(train_config.checkpoint_path)
    # %%
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config.checkpoint_path,
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=5,
        mode="min",
    )
    # data = PartitionPerEpochDataModule(train_config.batch_size, data_config)
    train_loader, val_loader = load_loader(train_config,data_config)
    # %%

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator="ddp",
        max_epochs=train_config.epochs,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=16, 
        amp_backend="native",
        profiler="simple"
    )

    # %%
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()