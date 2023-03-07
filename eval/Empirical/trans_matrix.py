import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import attack_whole_dataset

from utils.Empirical.datasets import DATASETS, get_dataset

from utils.Empirical.architectures import get_architecture
from models.ensemble import Ensemble ,Ensemble_logits , Ensemble_soft, Ensemble_vote,Ensemble_soft_baseline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--num-models', type=int, required=True)
parser.add_argument('--adv-eps', default=0.04, type=float)
parser.add_argument('--seed', default=[10011,777,12811], type=int)
parser.add_argument('--b', default=[100,128,128])
parser.add_argument('--mask', default=[1,1,1])
args = parser.parse_args()

def gen_plot(args, transmat):
    import itertools
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 3, step=1))
    plt.xticks(np.arange(0, 3, step=1))
    cmp = plt.get_cmap('Blues')
    plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
    plt.title("RandMod")
    # plt.colorbar()
    thresh = 50.0
    for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
        plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
                 horizontalalignment="center",
                 color="white" if transmat[i, j] > thresh else "black")

    plt.ylabel('Target model')
    plt.xlabel('Base model')
    outpath = "./outputs/Empirical/TransMatrix/"
    # plt.legend(frameon=False)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(outpath + "%s.pdf" % (args.outfile), bbox_inches='tight')

def main():
    models = []
    
    for i in range(args.num_models):
        if args.b[i]==0 :
            checkpoint = torch.load('/home/hrm/zwl/Transferability-Reduced/checkpoint.pth.tar.0')
            model = get_architecture(checkpoint["arch"], args.dataset)
            model = nn.DataParallel(model).cuda()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models.append(model)
        else:
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
    ensemble = Ensemble_soft_baseline(models,args.seed,args.b,args.mask)
    # ensemble.eval()
    print(len(models))
    print ('Model loaded')

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "cifar10")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128,
                             num_workers=4, pin_memory=pin_memory)

    trans = np.zeros((3, 3))

    adv = []
    loss_fn = nn.CrossEntropyLoss()

    for i in range(len(models)):
        curmodel = models[i]
        adversary = LinfPGDAttack(
            curmodel, loss_fn=loss_fn, eps=args.adv_eps,
            nb_iter=100, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
        adv.append(adversary)

    # for i in range(len(models)):
    #     test_iter = tqdm(test_loader, desc='Batch', leave=False, position=2)
    #     _, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
    #     for j in range(len(models)):
    #         right = 0
    #         for r in range((_.size(0) - 1) // 200 + 1):
    #             inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
    #             y = label[r * 200: min((r + 1) * 200, _.size(0))]
    #             y_pred = pred[r * 200: min((r + 1) * 200, _.size(0))]
    #             __ = adv[j].predict(inputc)
    #             output = (__).max(1, keepdim=False)[1]
    #             # trans[i][j] += (output == y).sum().item()
    #             right += (y == y_pred).sum().item()
    #             trans[i][j] += ((y == y_pred) & ((output != y))).sum().item()
    #         trans[i][j] /= right

    for i in range(len(models)):
        test_iter = tqdm(test_loader, desc='Batch', leave=False, position=2)
        _, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
        origin_acc = 0
        transfer_acc = 0
        for j in range(len(models)):
            right = 0
            for r in range((_.size(0) - 1) // 200 + 1):
                inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
                y = label[r * 200: min((r + 1) * 200, _.size(0))]
                y_pred =  pred[r * 200: min((r + 1) * 200, _.size(0))]
                y_advpred =  advpred[r * 200: min((r + 1) * 200, _.size(0))]
                __ = adv[j].predict(inputc)
                output = (__).max(1, keepdim=False)[1]
                ## 预测正确
                # origin_acc = (y_0 == y).sum().item()
                ## 攻击后预测正确
                # transfer_acc = (output == y).sum().item()
                # trans[i][j] += origin_acc - transfer_acc 
                trans[i][j] += (output == y).sum().item()
                # 基模型预测正确 且（ ‘默认’ ,攻击自己成功） 且（攻击j模型成功） 
                # right += (y == y_pred).sum().item()
                # trans[i][j] += ((output != y)&(output == y_advpred)).sum().item()
            trans[i][j] /= len(label)
    
    trans = (1-trans) * 100.
    print(trans)
    gen_plot(args, trans)

if __name__ == "__main__":
    main()
