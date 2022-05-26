import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import copy
import models
import torch.nn.functional as F
import pandas as pd
# from torch.autograd.gradcheck import zero_gradients
import data.poison_cifar as poison
from torch.autograd import Variable
from autoaugment import CIFAR10Policy, ImageNetPolicy
from PIL import Image
from data_loader import *
import matplotlib.pyplot as plt
import resnet_cifar


parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')        ## 32 to 1024
parser.add_argument('--lr', type=float, default=0.05, help='the learning rate for mask optimization')   ## lr-rate (0.001 to 0.02)
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')  ## decrease it by print-every
parser.add_argument('--print-every', type=int, default=10, help='print results every few iterations')  ## Change it couple of times ()
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='save/purified_models/')
parser.add_argument('--gpuid', type=int, default=0, help='the transparency of the trigger pattern.')

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'Feature', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')

parser.add_argument('--isolation_model_root', type=str, default='./weight/ABL_results',
                    help='isolation model weights are saved here')
parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned dataest')
parser.add_argument('--poisoned_data_test_all2one', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy', help='random seed')
parser.add_argument('--poisoned_data_test_all2all', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy', help='random seed')

# Backdoor Attacks
parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='squareTrigger', choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=1, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=1, help='height of trigger pattern')

args = parser.parse_args()
args_dict = vars(args)
# print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device =  'cpu'
torch.cuda.set_device(args.gpuid)


def main():
    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),                       ## For Strong Augmentation'
        # CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_train_weak = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_none = transforms.ToTensor()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    ## Clean Test Loader 
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)
    
    ## Triggers 
    triggers = {'badnets': 'checkerboard_1corner',
                'CLB': 'fourCornerTrigger',
                'blend': 'gaussian_noise',
                'SIG': 'signalTrigger',
                'TrojanNet': 'trojanTrigger',
                'FC': 'gridTrigger',
                'benign': None}

    if args.poison_type == 'badnets':
        args.trigger_alpha = 0.6

    # args.inject_portion = args.poison_rate
    
    ## Step 1: create dataset -- clean val set, poisoned test set, and clean test set.
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    
    elif args.poison_type in ['badnets', 'blend']:
        trigger_type  = triggers[args.poison_type]
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        trigger_info  = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                        'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}

        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)                   ## To check how many of the poisonous sample is correctly classified to their "target labels"
        # poison_test = poison.add_predefined_trigger_cifar_true_label(data_set=clean_test, trigger_info=trigger_info)      ## To check how many of the poisonous sample is correctly classified to their "true labels"
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

    elif args.poison_type in ['Dynamic', 'Feature']:
        transform_test = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        if args.target_type =='all2one':
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2one, allow_pickle=True), transform = None)
        else:
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2all, allow_pickle=True), transform = None)

        poison_test_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=False)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None

    elif args.poison_type in ['SIG', 'TrojanNet', 'CLB', 'FC']:
        trigger_type      = triggers[args.poison_type]
        args.trigger_type = trigger_type        

        ## SIG and CLB are Clean-label Attacks 
        if args.poison_type in ['SIG', 'CLB', 'FC']:
            args.target_type = 'cleanLabel'

        _, poison_test_loader = get_test_loader(args)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

        trigger_info = None

    elif args.poison_type == 'benign':
        trigger_info = None

                             ## Get the dataloader for Mask finetuning 
    orig_train        = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    _, clean_val_none = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    clean_val = clean_val_none                                                                                                   ## Removal using Clean val data
    # clean_val = poison.add_predefined_trigger_cifar_true_label(data_set=clean_val_none, trigger_info=trigger_info)             ## Removal using Triggered val data

    random_sampler    = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples =args.print_every * args.batch_size)
    clean_val_loader  = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)

                            ## Get the dataloader for Weight finetuning 
    # orig_train        = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train_weak)
    # _, clean_val_none = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
    #                                     perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    # clean_val = clean_val_none                                                                                                   ## Removal using Clean val data
    # # clean_val = poison.add_predefined_trigger_cifar_true_label(data_set=clean_val_none, trigger_info=trigger_info)             ## Removal using Triggered val data

    # random_sampler    = RandomSampler(data_source=clean_val, replacement=True,
    #                                num_samples =args.print_every * args.batch_size)
    # clean_val_loader  = DataLoader(clean_val, batch_size=args.batch_size,
    #                               shuffle=False, sampler=random_sampler, num_workers=0)

    ## Step 2: Load Model Checkpoints and Trigger Info
    state_dict = torch.load(args.checkpoint, map_location=device)
    if args.poison_type in ['Dynamic', 'Feature']:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']
        print("dynamic attack")

    # net = resnet_cifar.resnet18(num_classes =10, norm_layer=models.NoisyBatchNorm2d)                      ## FOr Umar Models
    net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d)                ## For Mask-finetuning 
    # net = getattr(models, args.arch)(num_classes=10)                                                  ## For Weight-finetuning
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.cuda()

    ## Loss Functions
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # ## Parameters 
    parameters  = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    # mask_params = [v for n, v in parameters if "linear" not in n]                                     ## Finetune conv Weights only
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.95)                            ## For Mask-finetuning
    # mask_optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.95)                     ## For Weight-finetuning

    # # Step 3: train backdoored models
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optimizer, nb_repeat, 0.04)

    ## Validate the model 
    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("Validation Accuracy of the Given Model:", cl_test_acc)
    print("Attack Success Rate of the Given Model:", po_test_acc)


        ## Losses and Accuracy ##
    clean_losses  = np.zeros(nb_repeat)
    poison_losses = np.zeros(nb_repeat)
    clean_accs    = np.zeros(nb_repeat)
    poison_accs   = np.zeros(nb_repeat)
    
    print("Number of Iterations:", nb_repeat)
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']

        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

        clean_losses[i]  = cl_test_loss
        poison_losses[i] = po_test_loss
        clean_accs[i]    = cl_test_acc
        poison_accs[i]   = po_test_acc

        ## Save numpy file
        np.savez(os.path.join(args.output_dir,'remove_loss_accuracy_adv_'+ args.poison_type + '_' + str(args.dataset) + '_.npz'), cl_loss = clean_losses, cl_test = clean_accs, po_loss = poison_losses, po_acc = poison_accs)
        model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) +'_' + str(args.inject_portion) + '_True_Label.pth'
        torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))

    save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values_pgd_strong.txt'))

                        ### Plot the loss and accuracy ####
    # clean_loss  = np.load(os.path.join(args.output_dir,'remove_loss_accuracy_'+ args.poison_type +  '_' + str(args.dataset) +'_.npz'))['cl_loss']
    # clean_acc = np.load(os.path.join(args.output_dir,'remove_loss_accuracy_'+ args.poison_type +  '_' + str(args.dataset) + '_.npz'))['cl_test']
    # poison_loss  =  np.load(os.path.join(args.output_dir,'remove_loss_accuracy_'+ args.poison_type + '_' + str(args.dataset) + '_.npz'))['po_loss']
    # poison_acc = np.load(os.path.join(args.output_dir,'remove_loss_accuracy_'+ args.poison_type +  '_' + str(args.dataset) +'_.npz'))['po_acc']

    # ## Plot and Save the Figures 
    # plot_epoch = 50
    # # fig, plt = plt1.subplots(figsize=(8,6))
    # # plt.rcParams['font.size'] = '20'
    # plt.figure(figsize=(8, 6))
    # x_axis = [i*10 for i in range(plot_epoch)]
    # plt.plot(x_axis, clean_loss[:plot_epoch], linewidth=4, color='b', label='Clean Loss')
    # plt.plot(x_axis, clean_acc[:plot_epoch], linewidth=4, color='tab:orange', label='Clean Acc')
    # plt.plot(x_axis, poison_loss[:plot_epoch], linewidth=4, color='m', label='Trigger Loss')
    # plt.plot(x_axis, poison_acc[:plot_epoch], linewidth=4, color='g', label='Attack Success Rate')

    # plt.ylabel('Loss/Accuracy', fontsize=20)
    # plt.xlabel('Number of Purification Epochs,' r'$E_p$' , fontsize=20)
    # plt.legend(prop={'size': 16})
    # plt.grid(linewidth = 0.3)
    # plt.savefig(os.path.join(args.output_dir, "Gini Coefficient"+ args.poison_type + str(args.inject_portion)+".pdf"), dpi=500, bbox_inches='tight')
    # plt.show()


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
            # print(k, v.size(), new_state_dict[k].size())
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.01, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def mask_train(model, criterion,mask_opt, data_loader):
    model.train()

    total_correct = 0
    total_loss    = 0.0
    nb_samples    = 0

    for i, (images, labels) in enumerate(data_loader):
        nb_samples += images.size(0)
        clean_images, labels = images.cuda(), labels.cuda()
        mask_opt.zero_grad()
   
        output_clean = model(clean_images)
        loss_nat     = criterion(output_clean, labels)

        tot_loss = loss_nat 
        
        tot_loss.backward()
        mask_opt.step()
        clip_mask(model)

        ## Claculate the train accuracy 
        pred = torch.max(output_clean,1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += tot_loss.item()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def finetune_train(model, criterion, mask_opt, data_loader):
    model.train()

    total_correct = 0
    total_loss    = 0.0
    nb_samples    = 0

    for i, (images, labels) in enumerate(data_loader):
        nb_samples += images.size(0)
        clean_images, labels = images.cuda(), labels.cuda()
        mask_opt.zero_grad()
   
        output_clean = model(clean_images)
        tot_loss     = criterion(output_clean, labels)
                    
        tot_loss.backward()
        mask_opt.step()

        pred = torch.max(output_clean,1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += tot_loss.item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            # print(labels, output.size())
            total_loss += criterion(output, labels).item()
            pred = torch.max(output,1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
