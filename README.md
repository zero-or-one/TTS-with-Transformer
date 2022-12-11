# TTS-with-Transformer
CS492(I) Final Project

Implementation of [Transformer TTS](https://arxiv.org/abs/1809.08895)  and [Multispeech Transformer TTS](https://arxiv.org/abs/2006.04664)

### Data
Train dataset is open sourced [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) 

### Checkpoints
Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1P9sMVEvwurFmICi4NbEg_97mxToE5cyy?usp=sharing) and place them to checkpoint directory

### Run
Specihy the needed parameters in hparams.py. Install requirements from requirements.txt
```
python3 infer.py --text "Text to convert into speech by this model."
```

## Reference code
* Transformer Network: https://github.com/soobinseo/Transformer-TTS
* Tacotron Network: https://github.com/keithito/tacotron