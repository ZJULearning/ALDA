import argparse
import os
import os.path as osp

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList, ImageList_label
from torch.autograd import Variable
import random
import math

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    dataset = loader['test']
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(dataset[i]) for i in range(10)]
            for i in range(len(dataset[0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    feature, predict_out = model(inputs[j])
                    predict_out = nn.Softmax(dim=1)(predict_out)
                    outputs.append(predict_out)
                outputs = sum(outputs) / 10
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(dataset)
            for i in range(len(dataset)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feature, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def image_label(loader, model, threshold=0.9, out_dir=None):
    # save the pseudo_label
    out_path = osp.join(out_dir, "pseudo_label.txt")
    print("Pseudo Labeling to ", out_path)
    iter_label = iter(loader["target_label"])
    with torch.no_grad():
        with open(out_path, 'w') as f:
            for i in range(len(loader['target_label'])):
                inputs, labels, paths = iter_label.next()
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                softmax_outputs = nn.Softmax(dim=1)(outputs)
                maxpred, pseudo_labels = torch.max(softmax_outputs, dim=1)
                pseudo_labels[maxpred < threshold] = -1
                for (path, label) in zip(paths, pseudo_labels):
                    f.write(path+' '+str(label.item())+'\n')
    return out_path

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    source_list = ['.'+i for i in open(data_config["source"]["list_path"]).readlines()]
    target_list = ['.'+i for i in open(data_config["target"]["list_path"]).readlines()]

    dsets["source"] = ImageList(source_list, \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=config['args'].num_worker, drop_last=True)
    dsets["target"] = ImageList(target_list, \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=config['args'].num_worker, drop_last=True)
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
            dsets["test"] = [ImageList(test_list, \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=config['args'].num_worker) for dset in dsets['test']]
    else:
        test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
        dsets["test"] = ImageList(test_list, \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=config['args'].num_worker)
    
    dsets["target_label"] = ImageList_label(target_list, \
                            transform=prep_dict["target"])
    dset_loaders["target_label"] = DataLoader(dsets["target_label"], batch_size=test_bs, \
            shuffle=False, num_workers=config['args'].num_worker, drop_last=False)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    if config["restore_path"]:
        checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))["base_network"]
        ckp = {}
        for k, v in checkpoint.items():
            if "module" in k:
                ckp[k.split("module.")[-1]] = v
            else:
                ckp[k] = v
        base_network.load_state_dict(ckp)
        log_str = "successfully restore from {}".format(osp.join(config["restore_path"], "best_model.pth"))
        config["out_file"].write(log_str+"\n")
        config["out_file"].flush()
        print(log_str)

    ## add additional network for some methods
    if "ALDA" in args.method:
        ad_net = network.Multi_AdversarialNetwork(base_network.output_num(), 1024, class_num)
    else:
        ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in range(len(gpus))])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(len(gpus))])
        
    loss_params = config["loss"]
    high = loss_params["trade_off"]
    begin_label = False
    writer = SummaryWriter(config["output_path"])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    loss_value = 0
    loss_adv_value = 0
    loss_correct_value = 0
    for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"]-1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = base_network #nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_step = i
                best_acc = temp_acc
                best_model = temp_model
                checkpoint = {"base_network": best_model.state_dict(), "ad_net": ad_net.state_dict()}
                torch.save(checkpoint, osp.join(config["output_path"], "best_model.pth"))
                print("\n##########     save the best model.    #############\n")
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            writer.add_scalar('precision', temp_acc, i)
            print(log_str)

            print("adv_loss: {:.3f} correct_loss: {:.3f} class_loss: {:.3f}".format(loss_adv_value, loss_correct_value, loss_value))
            loss_value = 0
            loss_adv_value = 0
            loss_correct_value = 0

            #show val result on tensorboard
            images_inv = prep.inv_preprocess(inputs_source.clone().cpu(), 3)
            for index, img in enumerate(images_inv):
                writer.add_image(str(index)+'/Images', img, i)
            
        # save the pseudo_label
        if 'PseudoLabel' in config['method'] and (i % config["label_interval"] == config["label_interval"]-1):
            base_network.train(False)
            pseudo_label_list = image_label(dset_loaders, base_network, threshold=config['threshold'], \
                                out_dir=config["output_path"])
            dsets["target"] = ImageList(open(pseudo_label_list).readlines(), \
                                transform=prep_dict["target"])
            dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                    shuffle=True, num_workers=config['args'].num_worker, drop_last=True)
            iter_target = iter(dset_loaders["target"]) # replace the target dataloader with Pseudo_Label dataloader
            begin_label = True

        if i > config["stop_step"]:
            log_str = "method {}, iter: {:05d}, precision: {:.5f}".format(config["output_path"], best_step, best_acc)
            config["final_log"].write(log_str+"\n")
            config["final_log"].flush()
            break
                 
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
        features_source, outputs_source = base_network(inputs_source)
        if args.source_detach:
            features_source = features_source.detach()
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        loss_params["trade_off"] = network.calc_coeff(i, high=high) #if i > 500 else 0.0
        transfer_loss = 0.0
        if 'DANN' in config['method']:
            transfer_loss = loss.DANN(features, ad_net)
        elif "ALDA" in config['method']:
            ad_out = ad_net(features)
            adv_loss, reg_loss, correct_loss = loss.ALDA_loss(ad_out, labels_source, softmax_out,
                        weight_type=config['args'].weight_type, threshold=config['threshold'])
            # whether add the corrected self-training loss
            if "nocorrect" in config['args'].loss_type:
                transfer_loss = adv_loss
            else:
                transfer_loss = config['args'].adv_weight * adv_loss + config['args'].adv_weight * loss_params["trade_off"] * correct_loss
            # reg_loss is only backward to the discriminator
            if "noreg" not in config['args'].loss_type:
                for param in base_network.parameters():
                    param.requires_grad = False
                reg_loss.backward(retain_graph=True)
                for param in base_network.parameters():
                    param.requires_grad = True
        # on-line self-training
        elif 'SelfTraining' in config['method']:
            transfer_loss += loss_params["trade_off"] * loss.SelfTraining_loss(outputs, softmax_out, config['threshold'])
        # off-line self-training
        elif 'PseudoLabel' in config['method']:
            labels_target = labels_target.cuda()
            if begin_label:
                transfer_loss += loss_params["trade_off"] * nn.CrossEntropyLoss(ignore_index=-1)(outputs_target, labels_target)
            else:
                transfer_loss += 0.0 * nn.CrossEntropyLoss(ignore_index=-1)(outputs_target, labels_target)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        loss_value += classifier_loss.item() / config["test_interval"]
        loss_adv_value += adv_loss.item() / config["test_interval"]
        loss_correct_value += correct_loss.item() / config["test_interval"]            
        total_loss = classifier_loss + transfer_loss
        total_loss.backward()
        optimizer.step()
    checkpoint = {"base_network": temp_model.state_dict(), "ad_net": ad_net.state_dict()}
    torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    return best_acc

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='ALDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--restore_dir', type=str, default=None, help="restore directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="trade off between supervised loss and self-training loss")
    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--cos_dist', type=str2bool, default=False, help="the classifier uses cosine similarity.")
    parser.add_argument('--threshold', default=0.9, type=float, help="threshold of pseudo labels")
    parser.add_argument('--label_interval', type=int, default=200, help="interval of two continuous pseudo label phase")
    parser.add_argument('--stop_step', type=int, default=0, help="stop steps")
    parser.add_argument('--final_log', type=str, default=None, help="final_log file")
    parser.add_argument('--weight_type', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--test_10crop', type=str2bool, default=True)
    parser.add_argument('--adv_weight', type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument('--source_detach', default=False, type=str2bool, help="detach source feature from the adversarial learning")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=True

    # train config
    config = {}
    config['args'] = args
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    config["restore_path"] = "snapshot/" + args.restore_dir if args.restore_dir else None
    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil
        shutil.rmtree(config["output_path"], ignore_errors=True)
    os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    if len(config['gpu'].split(','))>1:
        args.batch_size = 32*len(config['gpu'].split(','))
        print("gpus:{}, batch size:{}".format(config['gpu'], args.batch_size))

    config["prep"] = {"test_10crop":args.test_10crop, 'params':{"resize_size":256, "crop_size":224}}
    config["loss"] = {"trade_off":args.trade_off}
    if "ResNet" in args.net:
        net = network.ResNetFc
        config["network"] = {"name":net, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":512, "new_cls":True, 
            "cos_dist":args.cos_dist} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
            args.stop_step = 20000
        else:
            config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
        args.stop_step = 20000
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset has not been implemented.')
    if args.lr != 0.001:
        config["optimizer"]["lr_param"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["gamma"] = 0.001
    config["out_file"].write(str(config))
    config["out_file"].flush()
    config["threshold"] = args.threshold
    config["label_interval"] = args.label_interval
    if args.stop_step == 0:
        config["stop_step"] = 10000
    else:
        config["stop_step"] = args.stop_step
    if args.final_log is None:
        config["final_log"] = open('log.txt', "a")
    else:
        config["final_log"] = open(args.final_log, "a")
    train(config)
