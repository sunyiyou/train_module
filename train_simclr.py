# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from sklearn.model_selection import cross_val_score

import data
import torchvision

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')

parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='resnet18-simclr', type=str, help='model architecture')
parser.add_argument('--name', default='resnet18-simclr', type=str, help='name of experiment')
parser.add_argument('--inference-method', default='', type=str, help='')
parser.add_argument('--method', default='', type=str, help='odin mahalanobis')
parser.add_argument('--p', default=0, type=int, help='sparsity level')
parser.add_argument(
    "--training-mode", type=str, default="SimCLR", choices=("SimCLR", "SupCon", "SupCE")
)
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to rungit ')
parser.add_argument('--save-epoch', default=25, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=256, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               , type=float,
                    help='initial learning rate')
parser.add_argument('--beta', default=0.5, type=float,
                    help='')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument("--temperature", type=float, default=0.5)


parser.add_argument("--warmup", action="store_true")

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()
if args.name is None:
    args.name = args.model_arch
state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
np.random.seed(1)



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

DATASET_PATH = {
    'imagenet': '/home/sunyiyou/dataset/imagenet',
}

def main():

    kwargs = {'num_workers': 1, 'pin_memory': True}
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = TwoCropTransform(transform_train)

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        # lr_schedule=[100, 150, 222]
        num_classes = 100
    elif args.in_dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(DATASET_PATH['imagenet'], 'train'), transform=TwoCropTransform(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]))),
            batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(DATASET_PATH['imagenet'], 'val'), transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True)

        num_classes = 1000


    if args.in_dataset == "imagenet":
        if args.model_arch.find('resnet50') > -1:
            from models.resnet_ss import resnet50
            model = resnet50(num_classes=num_classes)
            state_dict = torch.load('checkpoints/supcon.pth')['model']
            state_dict = {key.replace("module.", "").replace("encoder.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
    else:
        from models.resnet_ss import resnet18_cifar, resnet34_cifar, resnet50_cifar, resnet101_cifar
        if args.model_arch.find('resnet18') > -1:
            model = resnet18_cifar(num_classes=num_classes, p=args.p, method=args.inference_method)
        elif args.model_arch.find('resnet50') > -1:
            model = resnet50_cifar(num_classes=num_classes, p=args.p, method=args.inference_method)
        elif args.model_arch.find('resnet34') > -1:
            model = resnet34_cifar(num_classes=num_classes, p=args.p, method=args.inference_method)
        elif args.model_arch.find('resnet101') > -1:
            model = resnet101_cifar(num_classes=num_classes, p=args.p, method=args.inference_method)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # args.start_epoch = checkpoint['epoch']
                state_dict = checkpoint['state_dict']
                state_dict = {key.replace("fc.", "head."): value for key, value in
                              state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                assert False, "=> no checkpoint found at '{}'".format(args.resume)


    if torch.cuda.device_count() > 1:
        import apex
        model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True


    criterion = SupConLoss(temperature=args.temperature).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    if args.warmup:
        wamrup_epochs = 10
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs * len(train_loader),
        )
        for epoch in range(wamrup_epochs):
            train(train_loader, model, criterion, optimizer, epoch)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, lr_schedule)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, lr_scheduler=lr_scheduler)

        # evaluate on validation set
        validate(val_loader, model)
        knn(val_loader, model, criterion, epoch)


        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    sup_losses = AverageMeter()
    ce_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    CELoss = nn.CrossEntropyLoss().cuda()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()

        batch_size = len(input[0])
        input = torch.cat([input[0], input[1]], dim=0).cuda()

        supfeat_output, ce_output = model(input, out_type="supce")
        f1, f2 = torch.split(supfeat_output, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if args.training_mode == "SupCon":
            sup_loss = criterion(features, target)
            loss = sup_loss
        elif args.training_mode == "SupCE":
            ce_loss = CELoss(ce_output, target.repeat(2).cuda())
            sup_loss = criterion(features, target)
            loss = 1. * sup_loss + args.beta * ce_loss
            ce_losses.update(ce_loss.data, batch_size)
            nat_prec1 = accuracy(ce_output.data, target.repeat(2).cuda(), topk=(1,))[0]
            nat_top1.update(nat_prec1, batch_size)
        elif args.training_mode == "SimCLR":
            sup_loss = criterion(features)
            loss = sup_loss
        else:
            raise ValueError("training mode not supported")

        # measure accuracy and record loss
        sup_losses.update(sup_loss.data, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            if args.training_mode == "SupCE":
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'SupLoss {sup_loss.val:.4f} ({sup_loss.avg:.4f})\t'
                      'CELoss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    sup_loss=sup_losses, ce_loss=ce_losses, top1=nat_top1))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=sup_losses))


def knn(val_loader, model, criterion, epoch):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].cuda(), data[1]
            output = model(images).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


def validate(val_loader, model):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    CELoss = nn.CrossEntropyLoss().cuda()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        # compute output
        output = model(input, out_type='ce')
        loss = CELoss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    # lr = args.lr * (0.1 ** (epoch // 60)) * (0.1 ** (epoch // 80))
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()
