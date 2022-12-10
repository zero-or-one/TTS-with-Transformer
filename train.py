import os
import torch
import torch.nn as nn
from utils import adjust_learning_rate


def train_epoch(model, optimizer, loss_fun, train_loader, logger, step, epoch, learning_rate):
    model.train()
    avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss = 0, 0, 0, 0, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in train_loader:
        step += 1
        if step < 400000:
            adjust_learning_rate(optimizer, learning_rate, step)
            
        text_input, mel, mel_input, pos_text, pos_mel, text_len = data
        
        stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1).unsqueeze(-1)
        
        text_input = text_input.to(device)
        mel = mel.to(device)
        mel_input = mel_input.to(device)
        pos_text = pos_text.to(device)
        pos_mel = pos_mel.to(device)
        
        mel_pred, postnet_pred, stop_preds, attn = model.forward(text_input, mel_input, text_len)

        loss, mel_loss, postnet_loss, stop_loss, att_loss = loss_fun(mel_pred, mel, postnet_pred, stop_preds, stop_tokens, attn)

        avg_loss += (loss.item())
        avg_mel_loss += (mel_loss.item())
        avg_postnet_loss += (postnet_loss.item())
        avg_stop_loss += (stop_loss.item())
        avg_att_loss += (att_loss)

        logger.log_alphas(model.module.encoder.pos_enc.alpha.data, \
            model.module.decoder.pos_enc.alpha.data, step)

        optimizer.zero_grad()
        # Calculate gradients
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        
        # Update weights
        optimizer.step()

        print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))

    avg_loss /= len(train_loader)
    return avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss


def validate(model, loss_fun, valid_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss = 0, 0, 0, 0, 0
    for data in valid_loader:

        text_input, mel, mel_input, pos_text, pos_mel, text_len = data
        stop_tokens = torch.abs(pos_mel.ne(0).type(torch.float) - 1).unsqueeze(-1)
        
        text_input = text_input.to(device)
        mel = mel.to(device)
        mel_input = mel_input.to(device)
        pos_text = pos_text.to(device)
        pos_mel = pos_mel.to(device)
        
        mel_pred, postnet_pred, stop_preds, attn = model.forward(text_input, mel_input, text_len)
        loss, mel_loss, postnet_loss, stop_loss, att_loss = loss_fun(mel_pred, mel, postnet_pred, stop_preds, stop_tokens, attn)

        avg_loss += (loss.item())
        avg_mel_loss += (mel_loss.item())
        avg_postnet_loss += (postnet_loss.item())
        avg_stop_loss += (stop_loss.item())
        avg_att_loss += (att_loss)

    avg_loss /= len(valid_loader)
    return avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss


def train(model, optimizer, loss_fun, train_loader, valid_loader, logger, hp, start_epoch=0):
    step = 0

    print("Training Started from Epoch {}".format(start_epoch))
    for epoch in range(start_epoch, hp.epochs):
        avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss \
             = train_epoch(model, optimizer, loss_fun, train_loader, logger, step, epoch, hp.learning_rate)
        step += 1
        logger.log_training(avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, epoch)

        
        if epoch % hp.save_interval == 0:
            avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, avg_att_loss \
                 = validate(model, loss_fun, valid_loader)

            logger.log_validation(avg_loss, avg_mel_loss, avg_postnet_loss, avg_stop_loss, epoch)
            torch.save(model.state_dict(), os.path.join(hp.checkpoint_path, 'checkpoint_{}.pth'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(hp.checkpoint_path, 'opt_checkpoint_{}.pth'.format(epoch)))

    print("Training Completed")

            
            