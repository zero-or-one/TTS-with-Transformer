import argparse
import os
import torch

from logger import TensorboardLogger
from loss import TotalLoss
from model import TransformerTTS
from hparams import BaseHparams
from data import prepare_dataloaders
from utils import seed_everything
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--visible_gpus', type=str, default="1",
                        required=False, help='CUDA visible GPUs')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hparams = BaseHparams()
    seed_everything(hparams.seed)
    start_epoch = 0

    train_loader, valid_loader = prepare_dataloaders(hparams)
    print("Data loaded")
    
    logger = TensorboardLogger()
    loss_fun = TotalLoss(hparams)

    model = TransformerTTS(hparams).to(device)
    if args.n_gpus >  1:
        print("Using {} GPUs".format(args.n_gpus))
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    
    if args.checkpoint_path:
        print("Loading checkpoint from {}".format(args.checkpoint_path))
        model.load_state_dict(torch.load(args.checkpoint_path))
        optimizer.load_state_dict(torch.load('opt_' + args.checkpoint_path))
        start_epoch = int(args.checkpoint_path.split('_')[-1].split('.')[0])
    
    train(model, optimizer, loss_fun, train_loader, valid_loader, logger, hparams, start_epoch)
