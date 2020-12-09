# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, top_choices_num=5, pos_weight=20, share_ensemble_help_pred_fc=True, force_all=False, use_norm=False, s=30):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
        else:
            s = 1
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])

        self.num_classes = num_classes

        self.top_choices_num = top_choices_num

        self.share_ensemble_help_pred_fc = share_ensemble_help_pred_fc
        self.layer3_feat = True

        ensemble_hidden_fc_output_dim = 16
        self.ensemble_help_pred_hidden_fcs = nn.ModuleList([nn.Linear((layer3_output_dim if self.layer3_feat else layer2_output_dim) * block.expansion, ensemble_hidden_fc_output_dim) for _ in range(self.num_experts - 1)])
        if self.share_ensemble_help_pred_fc:
            self.ensemble_help_pred_fc = nn.Linear(ensemble_hidden_fc_output_dim + self.top_choices_num, 1)
        else:
            self.ensemble_help_pred_fcs = nn.ModuleList([nn.Linear(ensemble_hidden_fc_output_dim + self.top_choices_num, 1) for _ in range(self.num_experts - 1)])

        self.pos_weight = pos_weight

        self.s = s

        self.force_all = force_all

        if not force_all:
            for name, param in self.named_parameters():
                if "ensemble_help_pred" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)
        self.feat = out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        out = (self.linears[ind])(out)
        out = out * self.s # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
        return out

    def pred_ensemble_help(self, input_part, i):
        feature, logits = input_part
        feature = F.adaptive_avg_pool2d(feature, (1, 1)).flatten(1)
        feature = feature / feature.norm(dim=1, keepdim=True)
        
        feature = F.relu((self.ensemble_help_pred_hidden_fcs[i])(feature))
        
        topk, _ = torch.topk(logits, k=self.top_choices_num, dim=1)
        confidence_input = torch.cat((topk, feature), dim=1)
        if self.share_ensemble_help_pred_fc:
            ensemble_help_pred = self.ensemble_help_pred_fc(confidence_input)
        else:
            ensemble_help_pred = (self.ensemble_help_pred_fcs[i])(confidence_input)
        return ensemble_help_pred

    def forward(self, x, target=None):
        out = F.relu(self.bn1(self.conv1(x)))
        shared_part = self.layer1(out)
        
        if target is not None: # training time
            output = shared_part.new_zeros((shared_part.size(0), self.num_classes))

            ensemble_help_preds = output.new_zeros((output.size(0), self.num_experts - 1), dtype=torch.float) 
            # first column: correctness of the first model, second: correctness of ensemble of the first and second, etc.
            correctness = output.new_zeros((output.size(0), self.num_experts), dtype=torch.bool) 

            loss = output.new_zeros((1,))
            for i in range(self.num_experts):
                output += self._separate_part(shared_part, i)
                correctness[:, i] = output.argmax(dim=1) == target # Or: just helpful, predict 1
                if i != self.num_experts - 1:
                    ensemble_help_preds[:, i] = self.pred_ensemble_help((self.feat, output / (i+1)), i).view((-1,))

            for i in range(self.num_experts - 1):
                # import ipdb; ipdb.set_trace()
                ensemble_help_target = (~correctness[:, i]) & correctness[:, i+1:].any(dim=1)
                ensemble_help_pred = ensemble_help_preds[:, i]
                
                # print("Helps ({}):".format(i+1), ensemble_help_target.sum().item() / ensemble_help_target.size(0))
                # print("Prediction ({}):".format(i+1), (torch.sigmoid(ensemble_help_pred) > 0.5).sum().item() / ensemble_help_target.size(0), (torch.sigmoid(ensemble_help_pred) > 0.3).sum().item() / ensemble_help_target.size(0))
                
                loss += F.binary_cross_entropy_with_logits(ensemble_help_pred, ensemble_help_target.float(), pos_weight=ensemble_help_pred.new_tensor([self.pos_weight]))
            
            # output with all ensembles
            return output / self.num_experts, loss / (self.num_experts - 1)
        else: # test time
            ensemble_next = shared_part.new_ones((shared_part.size(0),), dtype=torch.bool)
            num_experts_for_each_sample = shared_part.new_ones((shared_part.size(0), 1), dtype=torch.long)
            output = self._separate_part(shared_part, 0)
            for i in range(1, self.num_experts):
                ensemble_help_pred = self.pred_ensemble_help((self.feat, output[ensemble_next] / i), i-1).view((-1,))
                if not self.force_all:
                    ensemble_next[ensemble_next.clone()] = torch.sigmoid(ensemble_help_pred) > 0.5
                else:
                    """
                    if i == 1:
                        ensemble_next[ensemble_next.clone()] = torch.zeros(ensemble_next[ensemble_next.clone()].shape).uniform_(0, 1) > 0.4472
                    elif i == 2:
                        ensemble_next[ensemble_next.clone()] = torch.zeros(ensemble_next[ensemble_next.clone()].shape).uniform_(0, 1) > 0.1952
                    else:
                        print("Undefined pass ratio")
                    """
                print("Ensemble ({}):".format(i), ensemble_next.sum().item() / ensemble_next.size(0))
                
                if not ensemble_next.any():
                    break
                output[ensemble_next] += self._separate_part(shared_part[ensemble_next], i)
                num_experts_for_each_sample[ensemble_next] += 1
            
            return output / num_experts_for_each_sample, num_experts_for_each_sample

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()