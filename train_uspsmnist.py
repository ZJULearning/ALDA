import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
import os.path as osp
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import network
from tqdm import tqdm
import math

def train(args, model, ad_net, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, method):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    args.log_interval = num_iter
    high = args.trade_off
    
    loss_value = 0
    loss_target_value = 0
    for batch_idx in tqdm(range(num_iter), total=num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        features_source, outputs_source = model(data_source)
        features_target, outputs_target = model(data_target)
        feature = torch.cat((features_source, features_target), dim=0)
        output = torch.cat((outputs_source, outputs_target), dim=0)
        classifier_loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(output)
        if epoch > start_epoch:
            if method == 'DANN':
                transfer_loss = loss_func.DANN(feature, ad_net)
            elif method == "ALDA":
                ad_out = ad_net(feature)
                if label_source.size(0) != ad_out.size(0)//2:
                    continue
                adv_loss, reg_loss, correct_loss = loss_func.ALDA_loss(ad_out, label_source, softmax_output, 
                            weight_type=1, threshold=args.threshold)
                # whether add the corrected self-training loss
                if "nocorrect" in args.loss_type:
                    transfer_loss = adv_loss
                else:
                    transfer_loss = adv_loss + correct_loss
                # reg_loss is only backward to the discriminator
                if "noreg" not in args.loss_type:
                    for param in model.parameters():
                        param.requires_grad = False
                    reg_loss.backward(retain_graph=True)
                    for param in model.parameters():
                        param.requires_grad = True
            else:
                raise ValueError('Method cannot be recognized.')
            loss_target_value += transfer_loss.item() / args.log_interval
        else:
            transfer_loss = 0
        loss = classifier_loss + transfer_loss #loss_func.Square(softmax_output) + transfer_loss
        if math.isnan(loss.item()):
            raise AssertionError
        loss.backward()
        optimizer.step()
        if epoch > start_epoch:
            optimizer_ad.step()
        if batch_idx % args.log_interval == args.log_interval-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))
            print("transfer_loss: {:.3f} classifier_loss: {:.3f}".format(loss_target_value, loss_value))
            loss_value = 0
            loss_target_value = 0

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        feature, output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.data.cpu().max(1, keepdim=True)[1]
        correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='ALDA USPS2MNIST')
    parser.add_argument('method', type=str, default='ALDA', choices=['DANN', "ALDA"])
    parser.add_argument('--task', default='MNIST2USPS', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--gpu_id', type=str,
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--trade_off', type=float, default=1.0, help="trade_off")
    parser.add_argument('--start_epoch', type=int, default=0, help="begin adaptation after start_epoch")
    parser.add_argument('--threshold', default=0.9, type=float, help="threshold of pseudo labels")
    parser.add_argument('--output_dir', type=str, default=None, help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--cos_dist', type=str2bool, default=False, help="the classifier uses cosine similarity.")
    parser.add_argument('--num_worker', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.task == 'USPS2MNIST':
        source_list = './data/usps2mnist/usps_train.txt'
        target_list = './data/usps2mnist/mnist_train.txt'
        test_list = './data/usps2mnist/mnist_test.txt'
        start_epoch = 1
        decay_epoch = 6
    elif args.task == 'MNIST2USPS':
        source_list = './data/usps2mnist/mnist_train.txt'
        target_list = './data/usps2mnist/usps_train.txt'
        test_list = './data/usps2mnist/usps_test.txt'
        start_epoch = 1
        decay_epoch = 5
    else:
        raise Exception('task cannot be recognized!')

    source_list = open(source_list).readlines()
    target_list = open(target_list).readlines()
    test_list = open(test_list).readlines()

    train_loader = torch.utils.data.DataLoader(
        ImageList(source_list, transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last=True, pin_memory=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(target_list, transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(test_list, transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    model = network.USPS_EnsembNet()
    model = model.cuda()
    class_num = 10

    random_layer = None
    if args.method == "ALDA":
        ad_net = network.Multi_AdversarialNetwork(model.output_num(), 500, class_num)
    elif args.method == "DANN":
        ad_net = network.AdversarialNetwork(model.output_num(), 500)
    ad_net = ad_net.cuda()
    if args.task == 'USPS2MNIST':
        args.lr = 2e-4
    else:
        args.lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_ad = optim.Adam(ad_net.parameters(), lr=args.lr, weight_decay=0.0005)

    start_epoch = args.start_epoch
    if args.output_dir is None:
        args.output_dir = args.task.lower() +'_'+ args.method
    output_path = "snapshot/" + args.output_dir
    if os.path.exists(output_path):
        print("checkpoint dir exists, which will be removed")
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)

    for epoch in range(1, args.epochs + 1):
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, model, ad_net, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, args.method)
        test(args, model, test_loader)
        if epoch % 5 == 1:
            torch.save(model.state_dict(), osp.join(output_path, "epoch_{}.pth".format(epoch)))

if __name__ == '__main__':
    main()
