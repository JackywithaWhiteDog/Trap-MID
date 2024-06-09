from datetime import datetime
import json
import os
import sys
import time

import torch

import classify
import dataloader
import loss

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def init_dataloader(args, file_path, batch_size=64, mode="gan"):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    if args['dataset']['name'] == "celeba":
        data_set = dataloader.ImageFolder(args, file_path, mode)
    else:
        data_set = dataloader.GrayFolder(args, file_path, mode)

    data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    interval = time.time() - tf
    print('Initializing data loader took %ds' % interval)
    
    return data_set, data_loader

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def init_model(model_name, n_classes, pretrained_path):
    if model_name == "VGG16":
        net = classify.VGG16(n_classes)

    elif model_name == "FaceNet64":
        net = classify.FaceNet64(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(pretrained_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))

    elif model_name == "IR152":
        net = classify.IR152(n_classes)

        BACKBONE_RESUME_ROOT = os.path.join(pretrained_path, "Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        print("Loading Backbone Checkpoint ")
        load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))

    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    return net

def init_optimizer(model_args, parameters):
    optimizer_name = model_args.get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params=parameters,
                               lr=model_args['lr'], 
                               momentum=model_args['momentum'], 
                               weight_decay=model_args['weight_decay'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params=parameters,
                                lr=model_args['lr'],
                                weight_decay=model_args['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} not implemented.')

    if 'scheduler' in model_args:
        adjust_epochs = model_args['scheduler']['adjust_epochs']
        gamma = model_args['scheduler']['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)
    else:
        scheduler = None

    return optimizer, scheduler

def init_criterion(negls):
    if negls == 0:
        return torch.nn.CrossEntropyLoss().cuda()
    return loss.NegLSCrossEntropyLoss(negls)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, trap_info=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    if trap_info:
        for i, (key, value) in enumerate(trap_info.items()):
            print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')
