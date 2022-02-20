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

def adjust_learning_rate(optimizer, step_num, train_config):
    lr = train_config.lr * train_config.warmup_step**0.5 * min(step_num * train_config.warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(data_config, train_config):
    preprocessor = preprocess(data_config)
    train_dataset = LJDatasets(data_config, preprocessor, train=True)
    val_dataset = LJDatasets(data_config, preprocessor, train=False)
    global_step = 0

    m = nn.DataParallel(TransformerTTS(train_config, data_config).cuda())

    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=train_config.lr)

    pos_weight = torch.FloatTensor([5.]).cuda()
    writer = SummaryWriter()

    for epoch in range(train_config.epochs):

        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True,
                                  collate_fn=Transformer_Collator(preprocessor), drop_last=True, num_workers=16)
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d" % epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step, train_config)

            character, mel, mel_input, pos_text, pos_mel = data['text'], data['mel'], data['mel_input'], data['pos_text'], data['pos_mel']
            # character, mel, mel_input, pos_text, pos_mel, _ = data
            stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1)

            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()

            mel_out, postnet_out, stop_pred, enc_attn_list, mask_attn_list, enc_dec_attn_list = m(
                character, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_out, mel)
            post_mel_loss = nn.L1Loss()(postnet_out, mel)

            loss = mel_loss + post_mel_loss

            writer.add_scalars('training_loss', {
                'mel_loss': mel_loss,
                'post_mel_loss': post_mel_loss,

            }, global_step)

            writer.add_scalars('alphas', {
                'encoder_alpha': m.module.Text_Encoder.alpha.data,
                'decoder_alpha': m.module.Mel_Decoder.alpha.data,
            }, global_step)

            if global_step % train_config.image_step == 1:

                for i, prob in enumerate(enc_dec_attn_list):

                    num_h = prob.size(0)
                    for j in range(4):

                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0' %
                                         global_step, x, i*4+j)

                for i, prob in enumerate(enc_attn_list):
                    num_h = prob.size(0)

                    for j in range(4):

                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0' %
                                         global_step, x, i*4+j)

                for i, prob in enumerate(mask_attn_list):

                    num_h = prob.size(0)
                    for j in range(4):

                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0' %
                                         global_step, x, i*4+j)

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()

            if global_step % train_config.save_step == 0:
                torch.save({'model': m.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(train_config.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % global_step))


if __name__ == '__main__':
    data_config = DataConfig(
        root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
        train_csv='metadata_train.csv',
        val_csv='metadata_val.csv'
    )
    train_config = TrainConfig()
    main(data_config, train_config)
