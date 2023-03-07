
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Empirical.keydefense import Modular

class Trans(torch.autograd.Function):

    def forward(self, input_ ,seed,b,mask):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        self.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = Modular(seed,b,mask)(input_)               # relu就是截断负数，让所有负数等于0
        return output

    def backward(self, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu    
        input_, = self.saved_tensors # 把上面保存的input_输出
        # if self.needs_input_grad[0]:# 判断self.saved_tensors中的Variable是否需要进行反向求导计算梯度
        #     print('input_ need grad')
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input, None , None,None

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)

class Ensemble_soft_baseline(nn.Module):
    def __init__(self, models,seed,b,mask):
        super(Ensemble_soft_baseline, self).__init__()
        self.models = models
        self.seed = seed
        self.mask = mask
        self.b = b
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            i = 0
            for model in self.models :
                outputs += F.softmax( model(Trans.apply(x,self.seed[i],self.b[i],self.mask[i])), dim=-1)
                i = i + 1 
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](Trans.apply(x,self.seed[0],self.b[0],self.mask[0]))


class Ensemble_soft(nn.Module):
    def __init__(self, models,seed):
        super(Ensemble_soft, self).__init__()
        self.models = models
        self.seed = seed
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            i = 0
            for model in self.models:
                outputs += F.softmax( model(Trans.apply(x,self.seed[i])), dim=-1)
                i = i + 1 
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](Trans.apply(x,self.seed[0]))

class Ensemble_logits(nn.Module):
    # logits集成
    def __init__(self, models,seed):
        super(Ensemble_logits, self).__init__()
        self.models = models
        self.seed = seed
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            i = 0
            for model in self.models:
                outputs += model(Trans.apply(x,self.seed[i]))
                i = i + 1  
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](Trans.apply(x,self.seed[0]))


class Ensemble_vote(nn.Module):
    def __init__(self, models,seed):
        super(Ensemble_vote, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        t = 0.1
        if len(self.models) > 1:
            outputs = 0
            i = 0
            for model in self.models:
                outputs += F.softmax(model(Trans.apply(x,self.seed[i]))/t, dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](Trans.apply(x,self.seed[0]))