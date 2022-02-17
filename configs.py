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
    csv_name: str = 'metadata.csv'
    root_dir: str = './data/LJSpeech-1.1'
    symbols_len: int = 149
@dataclass
class TrainConfig():
    """
    Train Setting
    """
    hidden_size: int = 256
    embedding_size: int = 512
    n_iter: int = 60
    outputs_per_step: int = 1

    epochs: int = 10000
    lr: float = 0.001
    save_step: int = 2000
    image_step: int = 500
    batch_size: int = 32
    checkpoint_path: str = './checkpoint'
    sample_pathL: str = './samples'
