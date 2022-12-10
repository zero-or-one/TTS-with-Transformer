from tensorboardX import SummaryWriter

class TensorboardLogger():
    def __init__(self) -> None:
        self.logger = SummaryWriter()
        
    def log_training(self, loss, mel_loss, post_mel_loss, stop_loss, epoch):
        self.logger.add_scalars('loss',{
                'train_loss': loss,
                'train_mel_loss': mel_loss,
                'train_post_mel_loss': post_mel_loss,
                'train_stop_loss': stop_loss,
            }, epoch)

    def log_validation(self, loss, mel_loss, post_mel_loss, stop_loss, epoch):
        self.logger.add_scalars('loss',{
                'valid_loss': loss,
                'valid_mel_loss': mel_loss,
                'valid_post_mel_loss': post_mel_loss,
                'valid_stop_loss': stop_loss,
            }, epoch)

    def log_alphas(self, alpha_enc, alpha_dec, step):
        self.logger.add_scalars('alphas',{
                'encoder_alpha': alpha_enc,
                'decoder_alpha': alpha_dec,
            }, step)
            