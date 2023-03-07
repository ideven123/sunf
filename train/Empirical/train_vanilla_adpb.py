import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.Empirical.architectures import ARCHITECTURES
from utils.Empirical.datasets import DATASETS

from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, evaltrans
from utils.Empirical.datasets import get_dataset
from utils.Empirical.architectures import get_architecture
from train.Empirical.trainer import Naive_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N',
                    help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num-models', type=int, required=True)

parser.add_argument('--resume',default= 0 , action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--adveps', default=0.02, type=float)
# parser.add_argument('--seed', default=[777,1281,128051])
# parser.add_argument('--b', default=[128,128,128])
# parser.add_argument('--mask', default=[1,1,0.5])
parser.add_argument('--seed', default=[671441,671442,671443])
parser.add_argument('--b', default=[144,144,144])
parser.add_argument('--mask', default=[1,1,1,0.8,0.8,0.8])
args = parser.parse_args()

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}"
else:
    mode = f"vanilla"

args.outdir = f"/{args.dataset}/{mode}/"

args.epsilon /= 256.0

if (args.resume):
    args.outdir = "resume" + args.outdir
else:
    args.outdir = "scratch" + args.outdir

args.outdir = "logs/Empirical/" + args.outdir


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    copy_code(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = []
    for i in range(args.num_models):
        submodel = get_architecture(args.arch, args.dataset)
        submodel = nn.DataParallel(submodel)
        model.append(submodel)
    print("Model loaded")

    criterion = nn.CrossEntropyLoss().cuda()

    param = list(model[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(model[i].parameters()))

    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    writer = SummaryWriter(args.outdir)

    if (args.resume):
        base_classifier = "checkpoint.pth.tar.0"
        print(base_classifier)
        for i in range(args.num_models):
            checkpoint = torch.load(base_classifier)
            print("Load " + base_classifier)
            model[i].load_state_dict(checkpoint['state_dict'])
            model[i].train()
        print("Loaded...")

    
    for epoch in range(args.epochs):

        Naive_Trainer(args, train_loader, model, criterion, optimizer, epoch, device, writer)
        test(args,test_loader, model, criterion, epoch, device, writer)
        # evaltrans(args, test_loader, model, criterion, epoch, device, writer)

        scheduler.step(epoch)

        for i in range(args.num_models):
            model_path_i = model_path + ".seed" +"_%d" % (args.seed[i]) 
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model[i].state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path_i)


def get_adpb(train_loader):
    samp_cont = 0
    mu , sigma = 0,0
    for i, (inputs, targets) in enumerate(train_loader):
        # inputs, targets =inputs.to(device), targets.to(device)          
        batch_size = inputs.size(0)
        pixel = inputs.size(1)*inputs.size(2)*inputs.size(3)
        idx_list = np.arange(batch_size)

        samp_list = np.random.choice(idx_list, size= np.int64( np.ceil(0.02*batch_size) ), replace=False,p = None)
        samp_num = len(samp_list)
        
        # for idx in samp_list:
        curr_mu = torch.mean(inputs[samp_list])
        curr_sigma = torch.var(inputs[samp_list]) 
        
        mu = (mu*samp_cont + curr_mu*samp_num)/(samp_cont + samp_num)
        sigma = (sigma*samp_cont + curr_sigma*samp_num)/(samp_cont + samp_num)
        samp_cont = samp_cont + samp_num

    return mu , sigma


if __name__ == "__main__":
    main()
