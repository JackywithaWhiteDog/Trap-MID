from argparse import ArgumentParser
import os
import random

import numpy as np
import torch
import torch.nn as nn

import engine
import utils

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    model_name = args['dataset']['model_name']
    train_file = args['dataset']['train_file_path']
    test_file = args['dataset']['test_file_path']
    batch_size = args[model_name]['batch_size']
    _, trainloader = utils.init_dataloader(args, train_file, batch_size=batch_size, mode="train")
    _, testloader = utils.init_dataloader(args, test_file, batch_size=batch_size, mode="test")

    n_classes = args["dataset"]["n_classes"]
    pretrained_path = args['pretrained_path']
    net = utils.init_model(model_name, n_classes, pretrained_path)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args[model_name]['lr'], 
                                momentum=args[model_name]['momentum'], 
                                weight_decay=args[model_name]['weight_decay'])
    criterion = nn.CrossEntropyLoss().cuda()
    net = nn.DataParallel(net).cuda()

    channel = args["dataset"]["channel"]
    height = args["dataset"]["height"]
    width = args["dataset"]["width"]
    triggers = torch.rand((n_classes, channel, height, width))

    root_path = args['root_path']
    torch.save(triggers, os.path.join(root_path, "trigger.tar"))

    print("Start Training!")
    n_epochs = args[model_name]['epochs']
    best_model, best_acc, trapdoor_acc, triggers = engine.train(args, net, criterion, optimizer, trainloader, testloader, n_epochs, triggers)

    model_path = os.path.join(root_path, "target_ckp")
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}_{:.2f}_allclass.tar").format(model_name, best_acc, trapdoor_acc))

    if args['trapdoor']['optimized']:
        torch.save(triggers, os.path.join(root_path, "trigger.tar"))

if __name__ == '__main__':
    parser = ArgumentParser(description='Trap-MID training')
    parser.add_argument('--config', default='./config/classify_trap.json', type=str, help='Path to config file')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    namespace = parser.parse_args()

    if namespace.seed is not None:
        print(f'Set the random seed to be {namespace.seed}')
        set_random_seed(namespace.seed)

    file = namespace.config
    args = utils.load_json(json_file=file)
    model_name = args['dataset']['model_name']

    root_path = args['root_path']
    log_path = os.path.join(root_path, "target_logs")
    model_path = os.path.join(root_path, "target_ckp")
    os.makedirs(model_path, exist_ok=False)
    os.makedirs(log_path, exist_ok=False)

    if args['trapdoor']['optimized']:
        trigger_path = os.path.join(root_path, "triggers")
        os.makedirs(trigger_path, exist_ok=False)
        args['trigger_path'] = trigger_path

    args['trapdoor']['discriminator_loss'] &= args['trapdoor']['optimized']
    args['trapdoor']['discriminator_feat_loss'] &= args['trapdoor']['optimized']
    args['trapdoor']['discriminator_feat_model_loss'] &= args['trapdoor']['discriminator_feat_loss']

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    print(f'Load config file from {file}')

    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], trap_info=args["trapdoor"])

    main(args)