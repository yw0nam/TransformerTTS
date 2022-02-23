from dataclasses import dataclass

@dataclass
class DataConfig():
    """
    Data Settings
    """
    n_fft: int = 2048
    sr: int = 22050
    preemphasis: float = 0.97
    frame_shift: float = 0.0125  # seconds
    frame_length: float = 0.05  # seconds
    hop_length: int = 256       #(sr*frame_shift)
    win_length: int = 1024      #(sr*frame_length)
    n_mels: int  = 80  # Number of Mel banks to generate
    power: float = 1.2  # Exponent for amplifying the predicted magnitude
    min_level_db: int = -100
    ref_level_db: int = 20
    max_db: int = 100
    ref_db: int = 20
    cleaners: str = 'english_cleaners'
    train_csv: str = 'metadata.csv'
    val_csv: str = 'metadata.csv'
    root_dir: str = './data/LJSpeech-1.1'
    symbol_length: int = 149
    n_iter: int = 60
    train_samples: int = 12000
    
@dataclass
class TrainConfig():
    """
    Train Setting
    """
    bce_weight: int = 8
    hidden_size: int = 256
    n_head: int = 8
    embedding_size: int = 512
    n_layers: int = 5
    outputs_per_step: int = 1
    dropout_p: int = 0.1
    warmup_step: int = 4000
    warmup_ratio: float = 0.1
    epochs: int = 500
    lr: float = 0.001
    save_step: int = 5
    image_step: int = 2000
    batch_size: int = 64
    checkpoint_path: str = './my_engine/checkpoint/'
    model_save_path: str = './my_engine/best_model.bin'
    log_dir: str = './my_engine/tensorboard/'
    load_checkpoint: str = ""