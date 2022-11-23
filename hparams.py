from text import symbols
import numpy as np
from pathlib import Path

################################
# Experiment Parameters        #
################################
epochs=1000
iters_per_checkpoint=1000
seed=1234
dynamic_loss_scaling=True
fp16_run=False
fp16_opt_level='O1'
distributed_run=False
dist_backend="nccl"
dist_url="tcp://localhost"
dist_port=54325
cudnn_enabled=True
cudnn_benchmark=False
cudnn_deterministic=False
ignore_layers=['embedding.weight']
compute_alignments=False

################################
# Data Parameters             #
################################
csv_path = "../NeuralSpeech/Transformer-TTS/data/speaker_0/metadata.csv"
audio_dir = '../NeuralSpeech/Transformer-TTS/data/speaker_0/wavs/'
waveglow_path='/data2/sungjaecho/pretrained/waveglow_pretrained_700000.pt'

csv_data_paths={
    #'ljspeech':'./tacotron2_train/metadata/ljspeech.csv'
    #'emovdb':'./tacotron2_train/metadata/emovdb.csv'
    #'bc2013':'./tacotron2_train/metadata/bc2013.csv'
    #'ketts':'./tacotron2_train/metadata/ketts.csv'
    #'ketts2':'./tacotron2_train/metadata/ketts2.csv'
    #'nc':'./tacotron2_train/metadata/nc.csv'
    'kss':'./metadata/kss.csv'
    #'kss':'./metadata/20_speakers.csv'
    #'kss-w':'./metadata/20_speakers.csv'
}
shuffle=True
p_arpabet=1.0
cmudict_path='./text/cmu_dictionary'
text_cleaners='english_cleaners'

################################
# Audio Parameters             #
################################
max_wav_value=32768.0
sampling_rate=22050
n_fft=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin=0.0
mel_fmax=8000.0
f0_min=80
f0_max=880
harm_thresh=0.25
new_param=10
ref_db = 20
max_db = 100
preemphasis = 0.97
################################
# Model Parameters             #
################################
n_symbols=len(symbols)
symbols_embedding_dim=512

# (Text) Encoder parameters
encoder_kernel_size=5
encoder_n_convolutions=3
encoder_embedding_dim=512

# SpeakerEncoder parameters
speaker_embedding_dim=5

# LanguageEncoder parameters
max_languages=2
lang_embedding_dim=3

# EmotionEncoder parameters
emotion_embedding_dim=3
neutral_zero_vector=True

# SpeakerClassifier parameters
n_hidden_units=256
revgrad_lambda=1.0
revgrad_max_grad_norm=0.5

# Prosody predictor Parameters
prosody_dim=4
pp_lstm_hidden_dim=512
pp_opt_inputs = [''] # ['prev_global_prosody' 'AttRNN']

# Reference encoder
loss_ref_enc_weight=1.0
with_gst=True
#ref_enc_filters=[32 32 64 64 128 128]
#ref_enc_filter_size=[1 3] # [time_wise_stride freq_wise_stride]
#ref_enc_strides=[1 2] # [time_wise_stride freq_wise_stride]
#ref_enc_pad=[0 1] #[1 1]
ref_enc_gru_size=512
global_prosody_is_hidden=False

# Style Token Layer
token_embedding_size=512 # 128 in the paper 256
num_heads=8 
train_with_token_loss=True
gst_loss_weight=0.005

# Residual encoder parameters
res_en_out_dim=16
res_en_conv_kernels=512
res_en_conv_kernel_size=(33)
res_en_lstm_dim=256
std_lower_bound=np.exp(-2)
KLD_weight_scheduling='fixed' # ['fixed' 'pulse' 'cycle_linear']
## fixed KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
res_en_KLD_weight=1e-3
## pulse KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
KLD_weight_warm_up_step=15000
init_KLD_weight=0.001
KLD_weight_cof=0.002
## cyclic linear KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
cycle_KLDW_period=10000
cycle_KLDW_ratio=0.5
cycle_KLDW_min=0.0
cycle_KLDW_max=1e-5

# Decoder parameters
n_frames_per_step=1  # currently only 1 is supported
decoder_rnn_dim=1024
prenet_dim=256
max_decoder_steps=1000
gate_threshold=0.5
p_attention_dropout=0.1
p_decoder_dropout=0.1
style_to_attention_rnn=False#True
style_to_decoder_rnn=False#True
style_to_decoder_linear=False#True
style_to_encoder_output=False#False

# Attention parameters
attention_rnn_dim=1024
attention_dim=128

# MontonicAttention parameters
loss_att_means_weight=0.1
n_mean_units=1

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=31

# Mel-post processing network parameters
postnet_embedding_dim=512
postnet_kernel_size=5
postnet_n_convolutions=5

# Adversarial training with the speaker classfier
speaker_adv_weight=0.02
speaker_gradrev_lambda=1
speaker_gradrev_grad_max_norm=0.5

# Adversarial training with the emotion classfier
emotion_adv_weight=0.02
emotion_gradrev_lambda=1
emotion_gradrev_grad_max_norm=0.5

################################
# Optimization Hyperparameters #
################################
use_saved_learning_rate=True
learning_rate=1e-3
adam_batas=(0.9, 0.999)
adam_eps=1e-06
weight_decay=1e-6
lr_scheduling=False
lr_scheduling_start_iter=50000
lr_min=1e-5
grad_clip_thresh=1.0 # gradient clipping L2-norm
batch_size=16
mask_padding=True  # set model's padded outputs to padded values
freeze_pretrained=False
freeze_except_for=['nothing']
resampling_trainset_at_each_epoch=True