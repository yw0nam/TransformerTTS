# %%
import torch
from configs import DataConfig, TrainConfig
from utils import preprocess
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
# %%
data_config = DataConfig(
    root_dir="/home/spow12/data/TTS/LJSpeech-1.1/",
    train_csv='metadata_train.csv'
)
train_config = TrainConfig()
preprocessor = preprocess(data_config)
# %%
train_dataset = LJDatasets(data_config, preprocessor)
# %%
loader = DataLoader(train_dataset, batch_size=4,
                            collate_fn=Transformer_Collator(preprocessor), shuffle=False)
# %%
for i in loader:
    temp = i
    break
# %%
encoder_prenet = EncoderPrenet(train_config.embedding_size, train_config.hidden_size, data_config.symbols_len)
# %%
t = encoder_prenet(temp['text'])
# %%
t.size()
# %%
c_mask = temp['pos_text'].ne(0).type(torch.float)
mask = temp['pos_text'].eq(0).unsqueeze(1).repeat(1, temp['text'].size(1), 1)

# %%
mask.size()
# %%
