import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import copy
import sys
import os

from torch.optim import Optimizer
import time
import torch.nn.functional as F
import torch.autograd as autograd
# from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_
# from models.ensemble import Ensemble , Ensemble_soft , Ensemble_vote,Ensemble_logits,Ensemble_soft_baseline
# from utils.Empirical.keydefense import Modular

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.Empirical.architectures import ARCHITECTURES
from utils.Empirical.datasets import DATASETS

from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, evaltrans
from utils.Empirical.datasets import get_dataset
from utils.Empirical.architectures import get_architecture
from train.Empirical.trainer import Naive_Trainer , Model_Trainer
from utils.Empirical.keydefense import Modular
from models.ensemble import Ensemble , Ensemble_soft , Ensemble_vote,Ensemble_logits,Ensemble_soft_baseline

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('dataset', type=str, choices=DATASETS)
# parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--dataset', type=str, default = 'cifar10')
parser.add_argument('--arch', type=str, default = 'cifar_resnet20')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N',
                    help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lrx', '--learning-ratex', default=0.001, type=float,
                    help='initial learning rate', dest='lrx')
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
parser.add_argument('--num-models', type=int, default = 2)

parser.add_argument('--resume',default= 0 , action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--adveps', default=0.02, type=float)
# parser.add_argument('--seed', default=[777,1281,128051])
# parser.add_argument('--b', default=[128,128,128])
# parser.add_argument('--mask', default=[1,1,0.5])
parser.add_argument('--seed', default=[128021,128022,128023,128081,128082,128083])
parser.add_argument('--b', default=[128,128,128,128,128,128])
parser.add_argument('--mask', default=[0.2,0.2,0.2,0.8,0.8,0.8])
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
    train_loader0 = DataLoader(train_dataset, shuffle=False, batch_size=args.batch,
                              num_workers=0)
    train_loader1 = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    x_list = []
    y_list = []
    x0_list = []
    y0_list = []
    batch_num =np.int32(np.ceil(50000/args.batch))
    for i, (inputs, targets) in enumerate(train_loader0):
        inputs, targets =inputs.to(device),  targets.to(device)
        x0_list.append(inputs)
        y0_list.append(targets)
        x_list = copy.deepcopy(x0_list)
        y_list = copy.deepcopy(y0_list)

    # print(x_list,y_list)  
    model = []
    for i in range(args.num_models):
        submodel = get_architecture(args.arch, args.dataset)
        submodel = nn.DataParallel(submodel)
        model.append(submodel)
    print("Model loaded")

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_x =nn.CosineSimilarity().cuda()

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

        if epoch % 10 == 9:
            print("############update data############")
            update_x(args, x0_list,y0_list,x_list,y_list,batch_num, model, criterion_x, epoch, device, writer)
        
        #### ??????model0
        Naive_Trainer(args,  x0_list,y0_list, batch_num, model[0], criterion, optimizer, epoch, device, writer)
        #### ??????model1 ,  ?????????train_loader???????????????????????????

        Model_Trainer(args, x_list,y_list, batch_num, model[1], criterion, optimizer, epoch, device, writer)
        test(args,test_loader, model[1], criterion, epoch, device, writer)
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




def update_x(args, x0_list,y0_list,x_list,y_list, batch_num , models, criterion,epoch: int, device: torch.device, writer=None ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

	# for i in range(args.num_models):
	# 	models[i].train()
	# 	requires_grad_(models[i], True)
	# ## ??????modular?????????
	# end = time.time()
    
    ##### ??????loss function
    # optimizer = optim.SGD(x_list , lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)        
    loss_fn = nn.CrossEntropyLoss()
    adversary0 = LinfPGDAttack(
                models[0], loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
    adversary1 = LinfPGDAttack(
                models[0], loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
    
    # adv = adversary.perturb(data, label)
    # advpred = adversary.predict(adv)
    for i in range(batch_num):

        input,target = x_list[i], y_list[i]
        inputs,targets = x0_list[i], y0_list[i]
        
        batch_size = inputs.size(0)
        # if input == inputs :
        #     print(1)

        ### ??????????????????loss?????????
        # input.requires_grad = True
        grads = []

        optimizer = optim.SGD([input] , lr=args.lrx, momentum=args.momentum, weight_decay=args.weight_decay) 
        # ?????????
        # logits0 = models[0](inputs)
        logits1 = models[1](input)
        # loss = criterion(logits0, logits1)

        # ??????2
        adv0 =  adversary0.perturb(inputs, targets) - inputs
        adv1 = adversary1.perturb(input, target) - input
        loss = criterion(adv0, adv1)

        loss = torch.sum(loss)
        grad = autograd.grad(loss,input ,create_graph=True)[0]
		# grad = grad.flatten(start_dim=1)
		# grads.append(grad)
		# loss_std += loss

		# measure accuracy and record loss
        acc1, acc5 = accuracy(logits1, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.avg:.3f}\t'
					'Data {data_time.avg:.3f}\t'
					'Loss {loss.avg:.4f}\t'
					'Acc@1 {top1.avg:.3f}\t'
					'Acc@5 {top5.avg:.3f}'.format(
				epoch, i, batch_num, batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top5=top5))

    data_time.update(time.time() - end)
    print('Epoch: [{0}][{1}/{2}]\t'
					'Data {data_time.avg:.3f}\t'.format(
				epoch, i, batch_num,
				data_time=data_time))
		
    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)




if __name__ == "__main__":
    main()
