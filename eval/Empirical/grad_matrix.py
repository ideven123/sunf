import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import attack_whole_dataset

from utils.Empirical.datasets import DATASETS, get_dataset

from utils.Empirical.architectures import get_architecture
from models.ensemble import Ensemble ,Ensemble_logits , Ensemble_soft, Ensemble_vote , Trans
from utils.Empirical.utils_ensemble import AverageMeter
from utils.Empirical.keydefense import Modular

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--num-models', type=int, required=True)
parser.add_argument('--adv-eps', default=0.04, type=float)
parser.add_argument('--seed', default=[777,128071,128051], type=int)
parser.add_argument('--b', default=[128,128,128])
parser.add_argument('--mask', default=[1,0.7,0.5])
args = parser.parse_args()

def gen_plot(args, transmat):
    import itertools
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 3, step=1))
    plt.xticks(np.arange(0, 3, step=1))
    cmp = plt.get_cmap('Blues')
    plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
    plt.title("Transfer attack success rate")
    plt.colorbar()
    thresh = 50.0
    for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
        plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
                 horizontalalignment="center",
                 color="white" if transmat[i, j] > thresh else "black")

    plt.ylabel('Target model')
    plt.xlabel('Base model')
    outpath = "./outputs/Empirical/TransMatrix/"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(outpath + "%s.pdf" % (args.outfile), bbox_inches='tight')

def main():
    models = []
    for i in range(args.num_models):
        checkpoint = torch.load(args.base_classifier + "%d" % (args.seed[i]))
        model = get_architecture(checkpoint["arch"], args.dataset)
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models.append(model)

    # ensemble = Ensemble(models)
    # ensemble = Ensemble_soft(models,args.seed)
    # ensemble = Ensemble_logits(models,args.seed)
    # ensemble = Ensemble_vote(models,args.seed)
    # ensemble.eval()

    print ('Model loaded')
    cos01_losses = AverageMeter()   
    cos02_losses = AverageMeter()
    cos12_losses = AverageMeter()

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "cifar10")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128,
                             num_workers=4, pin_memory=pin_memory)

    trans = np.zeros((3, 3))

    adv = []
    loss_fn = nn.CrossEntropyLoss().cuda()

    cos01_max = -1
    cos02_max = -1
    cos12_max = -1
    for _, (inputs, targets) in enumerate(test_loader):
        # print(targets)
        targets = torch.zeros(size = [targets.size(0)],dtype = int)
        # print(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        # inputs.requires_grad = True

        grads = []
        for j in range(args.num_models):
            # logits = models[j](inputs) 
            In =  Modular(args.seed[j],args.b[j],args.mask[j])(inputs)           
            In.requires_grad = True
            logits = models[j](In)
            loss = loss_fn(logits, targets)
            
            # logits = models[j]( Modular(args.seed[j],args.b[j],args.mask[j])(inputs))
            # logits = models[j]( Trans.apply(inputs, args.seed[j]))
            loss = loss_fn(logits, targets)
            gradj = torch.autograd.grad(loss,  In , create_graph=True)[0]
            gradj = gradj.flatten(start_dim=1)
            grads.append(gradj)

        # cos01 = torch.abs( F.cosine_similarity(grads[0], grads[1])).mean()  
        # cos02 =  torch.abs( F.cosine_similarity(grads[0], grads[2])).mean() 
        # cos12 =  torch.abs( F.cosine_similarity(grads[1], grads[2])).mean() 

        # cos01 = F.cosine_similarity(grads[0], grads[1]).mean()  
        # cos02 = F.cosine_similarity(grads[0], grads[2]).mean() 
        # cos12 = F.cosine_similarity(grads[1], grads[2]).mean() 
        # cos01_losses.update(cos01.item(), batch_size)
        # cos02_losses.update(cos02.item(), batch_size)
        # cos12_losses.update(cos12.item(), batch_size)
        cos01 = F.cosine_similarity(grads[0], grads[1]).max()  
        cos02 = F.cosine_similarity(grads[0], grads[2]).max() 
        cos12 = F.cosine_similarity(grads[1], grads[2]).max()        
        cos01_max = max(cos01_max,cos01)
        cos02_max = max(cos02_max,cos02)
        cos12_max = max(cos12_max,cos12)
    print("cos01:",cos01_max , cos01_losses.count)
    print("cos02:",cos02_max, cos02_losses.count)
    print("cos12:",cos12_max , cos12_losses.count)
    # print("cos01:",cos01_losses.avg , cos01_losses.count)
    # print("cos02:",cos02_losses.avg , cos02_losses.count)
    # print("cos12:",cos12_losses.avg , cos12_losses.count)


if __name__ == "__main__":
    main()
