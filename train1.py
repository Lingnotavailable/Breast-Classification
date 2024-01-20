'''
Training script for Medical image data
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import medicel_data_test_utils
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, classification_report
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
from datetime import datetime
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import resnet18, ResNet18_Weights

torch.cuda.empty_cache()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Folder name
now = datetime.now()
folder = now.strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser(description='PyTorch Medical image Training')
# Datasets
parser.add_argument('-d', '--dataset', default='medicalSet', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=10
                    , type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--k', '--kfold', default=5, type=int,
                    metavar='K', help='Number of folds')                
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=56, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# get args from terminal and save in a dict
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset where only cifar10 and cifar100 are suitble
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'medicalSet','Dataset can only be cifar10 or cifar100.'

# Select and use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)  # python random seed generation
torch.manual_seed(args.manualSeed)  # cpu random seed generation
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)  # gpu random seed generation

best_acc = 0  # best test accuracy


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    top5 = AverageMeter()
    f1_score = AverageMeter()
    recall = AverageMeter()
    precision = AverageMeter()
    auc_roc = AverageMeter()

    total_target, total_output = torch.tensor([], dtype=torch.float32, device="cpu"), torch.tensor([], dtype=torch.float32, device="cpu")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            outputs = nn.functional.sigmoid(outputs)
            total_output = torch.cat([total_output, outputs.clone().cpu()])
            total_target = torch.cat([total_target, targets.clone().cpu()])
            loss = criterion(outputs, targets.unsqueeze(1).float())
            losses.update(loss.item(), inputs.size(0))

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
      #     accuracy=accuracy.avg,
      #       auc=auc_roc.avg,
      #       f1=f1_score.avg
        )
        bar.next()
    bar.finish()
    metrics, roc, cm = classification_report(total_output.data, total_target.data)
    
    if metrics['acc'] > best_acc:
        save_metrics(cm, roc)
        checkpoint_folder = os.path.join(args.checkpoint,os.path.join(args.arch,folder))
        plot_roc_curve(total_target.data, total_output.data, checkpoint_folder+'/ROC.jpg')
    return (losses.avg, metrics, roc)

def plot_roc_curve(y_true, y_scores, file_name):
    """
    Plots and saves the ROC curve.

    Parameters:
    y_true (torch.Tensor): True binary labels.
    y_scores (torch.Tensor): Predicted probabilities.
    file_name (str): The name of the file to save the plot.

    Returns:
    None, saves a plot to a file.
    """

    # Ensure the input is a numpy array
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(file_name)
    plt.close()


def save_metrics(cm, roc):

    # saving confusion matrix
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')

    checkpoint_folder = os.path.join(args.checkpoint,os.path.join(args.arch,folder))
    plt.savefig(checkpoint_folder+'/confusion_matrix.jpg')
    ax.remove()

    # saving roc curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(roc['fpr'], roc['tpr'], 'b', label = 'AUC = %0.2f' % roc['auc'])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(checkpoint_folder+'/ROC.jpg')


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    checkpoint_folder = os.path.join(args.checkpoint,os.path.join(args.arch,folder))
    os.makedirs(checkpoint_folder, exist_ok = True)
 

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # for medical data test
    transform_medical = transforms.Compose([
        transforms.Resize((224, 224)),

        # transforms.RandomHorizontalFlip(),
        # # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(num_output_channels=3),
        # transforms.CenterCrop(32),  # crop 224x224 
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    # -------------------

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        
    # for medical data test
    if args.dataset == 'medicalSet':
        dataloader = medicel_data_test_utils.medical_dataset.dataset
        num_classes = 1
    # --------------------
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        # model = models.__dict__[args.arch](
        #     cardinality=args.cardinality,
        #     num_classes=num_classes,
        #     depth=args.depth,
        #     widen_factor=args.widen_factor,
        #     dropRate=args.drop,
        # )
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=num_classes,
            baseWidth=args.widen_factor,
        )        
        
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            block_name=args.block_name,
       )

    elif args.arch.endswith('resnet50'):
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        num_features = model.fc.in_features 

        model.fc = nn.Linear(num_features, 1)
    elif args.arch.endswith('resnet18_sd'):
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_features = model.fc.in_features 
        model.fc = nn.Linear(num_features, 1)
        
    elif args.arch.endswith('resnet18'):
        model = resnet18(weights=None)
        checkpoint_path = '/home/jovyan/ling/SimCLR/checkpoint_0000.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # Remove 'module.' prefix and filter out the 'fc' layer from the checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}

        # Load the state dict into your model, excluding the 'fc' layer
        model.load_state_dict(state_dict, strict=False)
        num_features = 512
        model.fc = nn.Linear(num_features, 1)
    # Adjusting the fc layer to match the checkpoint structure
        #num_features = model.fc.in_features
        #model.fc = nn.Sequential(
        #    nn.Linear(num_features, 512),  # First linear layer from num_features to 512
        #    nn.ReLU(),                     # Assuming there might be an activation function; adjust as needed
        #    nn.Linear(512, 128)            # Second linear layer from 512 to 128
        #)
        #model.load_state_dict(checkpoint)
    # Now redefine the model.fc for binary classification
    # The final layer should output 1 feature for binary classification
        #model.fc = nn.Sequential(
        #    nn.Linear(num_features, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 1)
        #)

    elif args.arch.endswith('vgg19'):
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 1)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    #--------------------
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # criterion = nn.CrossEntropyLoss() 
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #--------------------
    splits = KFold(n_splits=args.k, shuffle=True, random_state=42)

    if args.dataset != 'medicalSet':
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

        mainset = data.ConcatDataset([trainset, testset])
    #for medical data test
    else:
        mainset = dataloader('data/images',
                          #  'data/label.csv',
                            transform=transform_medical)
    
    #--------------------
    total_f1 = AverageMeter()
    total_train_loss = AverageMeter()
    total_test_loss = AverageMeter()
    total_train_acc = AverageMeter()
    total_test_acc = AverageMeter()

    logger_reset = 1
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(mainset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = data.SubsetRandomSampler(train_idx)
        test_sampler = data.SubsetRandomSampler(val_idx)

        trainloader = data.DataLoader(mainset, batch_size=args.train_batch, num_workers=args.workers, sampler=train_sampler)
        testloader = data.DataLoader(mainset, batch_size=args.test_batch, num_workers=args.workers, sampler=test_sampler)


    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            # args.checkpoint = os.path.dirname(args.resume)
        checkpoint_folder = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_folder, 'log.txt'), title=title, resume=True)
    else:
            if logger_reset:
                logger = Logger(os.path.join(checkpoint_folder, 'log.txt'), title=title)
                logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'f1_Score'])
                logger_reset = 0

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, auc_score, roccurve = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_prec = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, metrics, roccurve = test(testloader, model, criterion, epoch, use_cuda)

        test_acc = metrics['acc']
        total_f1.update(metrics['f1'],1)
        total_train_loss.update(train_loss,1)
        total_test_loss.update(test_loss,1)
        total_train_acc.update(train_acc,1)
        total_test_acc.update(test_acc,1)
        
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint_folder)

        # append logger file
        logger.append([state['lr'], total_train_loss.avg, total_test_loss.avg, total_train_acc.avg, total_test_acc.avg, total_f1.avg])


    print('Best acc:')
    print(best_acc)
    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_folder, 'log.eps'))
    
def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precision = AverageMeter()
    accuracy = AverageMeter()
    f1_s = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        outputs = nn.functional.sigmoid(outputs)
        loss = criterion(outputs, targets.unsqueeze(1).float())

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
        metrics, roc, _= classification_report(outputs.data, targets.data)
        losses.update(loss.item(), inputs.size(0))
        precision.update(metrics['prec'], inputs.size(0))
        f1_s.update(metrics['f1'], inputs.size(0))
        accuracy.update(metrics['acc'], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} |f1: {f1:.4f}| accuracy: {accuracy:.4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            accuracy=accuracy.avg,
            f1=f1_s.avg
        )
        bar.next()
    bar.finish()
    return (losses.avg, accuracy.avg, precision.avg)

if __name__ == '__main__':
    main()
