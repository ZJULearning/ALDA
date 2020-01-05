import argparse
import os
import os.path as osp

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
import data_list
from data_list import ImageList, ImageList_label
from torch.autograd import Variable
import math

class Inspector():
    def __init__(self, num_class):
        self.num_class = num_class
        self.preds = []
        self.acc = []

    def add_batch(self, pred, label, ad_out=None):
        maxpred, argpred = torch.max(pred.data.cpu(), dim=1)
        sample = np.concatenate([maxpred.numpy().reshape(-1,1), (argpred==label).float().numpy().reshape(-1,1)], axis=1)
        self.preds.append(sample)

    def report(self):
        preds = np.concatenate(self.preds, axis=0)
        preds = np.array(sorted(preds, key=lambda x: x[0], reverse = True))
        num = len(preds)
        n_ = 0
        for i in range(1, 21):
            n = int(math.floor(num*0.05*i))
            acc_top = sum(preds[:n, 1]) / n
            acc = sum(preds[n_:n, 1]) / (n-n_)
            n_ = n
            print("{}%: maxprob {:.3f}, acc_top {:.3f} acc {:.3f};".format(5*i, preds[n-1][0], acc_top, acc), end="  ")
            if i % 2==0:
                print(" ")

class Eval():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.ignore_index = None
        self.synthia = True if num_class == 16 else False


    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Precision(self, out_16_13=False):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision
    
    def Print_Every_class_Eval(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / (1+self.confusion_matrix.sum(axis=1))
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        pas = []
        for ind_class in range(len(MPA)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            pas.append(pa)
        print(pas)
        print(np.nanmean(MPA))

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test_noise"])
            for i in range(len(loader['test_noise'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, predict.numpy().astype(int), all_label.numpy().astype(int)

def evaluate(config):
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

    source_list = open(data_config["source"]["list_path"]).readlines()
    target_list = open(data_config["target"]["list_path"]).readlines()

    dsets["source"] = ImageList(source_list, \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(target_list, \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
            dsets["test"] = [ImageList(test_list, \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        test_list = ['.'+i for i in open(data_config["test"]["list_path"]).readlines()]
        dsets["test"] = ImageList(test_list, \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    if config["restore_path"]:
        checkpoint = torch.load(osp.join(config["restore_path"], "best_model.pth"))
        base_network.load_state_dict(checkpoint)
        print("successfully restore from ", osp.join(config["restore_path"], "best_model.pth"))

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(len(gpus))])
        
    evaluater = Eval(class_num)

    base_network.train(False)
    temp_acc, predict, all_label = image_classification_test(dset_loaders, \
        base_network, test_10crop=prep_config["test_10crop"])
    print("arg acc:", temp_acc)
    evaluater.add_batch(predict, all_label)
    evaluater.Print_Every_class_Eval()
    evaluater.reset()
    return

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='ALDA', choices=['DANN', 'ALDA'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--restore_dir', type=str, default=None, help="restore directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="trade_off")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--cos_dist', default=False, type=str2bool, help="cos_dist")
    parser.add_argument('--epsilon', type=float, default=2.5)
    parser.add_argument('--stop_step', type=int, default=0, help="stop_step")
    parser.add_argument('--final_log', type=str, default=None, help="final_log file")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
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

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off}
    if "ResNet" in args.net:
        if args.AdaBN:
            net = network.ResNetFc_AdaBN
        else:
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
        if "DANN" in config['method']:
            config["optimizer"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 31 
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
    config["epsilon"] = args.epsilon
    if args.stop_step == 0:
        config["stop_step"] = 10000
    else:
        config["stop_step"] = args.stop_step
    if args.final_log is None:
        config["final_log"] = open('log.txt', "a")
    else:
        config["final_log"] = open(args.final_log, "a")
    evaluate(config)