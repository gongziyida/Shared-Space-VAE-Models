import os
import torch
from cub.experiment import CUB
from tensorboardX import SummaryWriter
from utils import utils
from utils.TBLogger import TBLogger
from tqdm import tqdm
from run_epochs import train, test

EXP_DIR = 'runs/CUB/poe/laplace_categorical/wrew/'
exp = CUB(flags)
exp.mm_vae.load_state_dict(torch.load(os.path.join(EXP_DIR, 'checkpoints/0044/mm_vae')))

flags = torch.load(os.path.join(EXP_DIR, 'flags.rar'))

writer = SummaryWriter(EXP_DIR)
tb_logger = TBLogger(exp.flags.str_experiment, writer)

exp.set_optimizer()
exp.optimizer

print('training epochs progress:')
for epoch in tqdm(range(45, exp.flags.end_epoch)):
    # utils.printProgressBar(epoch, exp.flags.end_epoch)
    # one epoch of training and testing
    train(epoch, exp, tb_logger)
    test(epoch, exp, tb_logger)
    # save checkpoints after every 5 epochs
    if (epoch + 1) % 5 == 0 or (epoch + 1) == exp.flags.end_epoch:
        dir_network_epoch = os.path.join(EXP_DIR, str(epoch).zfill(4))
        if not os.path.exists(dir_network_epoch):
            os.makedirs(dir_network_epoch)
        exp.mm_vae.save_networks()
        torch.save(exp.mm_vae.state_dict(),
                   os.path.join(dir_network_epoch, exp.flags.mm_vae_save))