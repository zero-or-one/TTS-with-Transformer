from text import symbols

class BaseHparams:
    def __init__(self):

        self.seed=13

        # Optimization Hyperparameters
        self.learning_rate=1e-3
        self.adam_batas=(0.9, 0.999)
        self.adam_eps=1e-06
        self.weight_decay=1e-6
        self.grad_clip_thresh=1.0
        self.batch_size=16
        self.mask_padding=True 

        # Data Parameters
        self.csv_path = "../NeuralSpeech/Transformer-TTS/data/speaker_0/metadata.csv"
        self.audio_dir = '../NeuralSpeech/Transformer-TTS/data/speaker_0/wavs/'
        self.waveglow_path='/data2/sungjaecho/pretrained/waveglow_pretrained_700000.pt'
        self.shuffle=True
        self.split_ratio=0.97
        self.text_cleaners='english_cleaners'

        # Audio Parameters 
        self.sampling_rate=22050
        self.n_fft=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mel_channels=80
        self.f0_min=80
        self.f0_max=880
        self.ref_db = 20
        self.max_db = 100
        self.preemphasis = 0.97

        # Training Parameters
        self.epochs=1000
        self.save_interval=5

        # Model Parameters 
        self.n_symbols=len(symbols)
        self.symbols_embedding_dim=256
        self.num_hidden = 256
        self.num_ffn = 256
        self.num_layers = 6
        self.num_outputs = 1
        self.num_heads = 8
        self.dropout = 0.1
        self.attn_dropout = 0.1
        