from copy import deepcopy
import os
import time

import kornia
import torch
import torch.nn as nn

import utils

def blend(img, key, alpha):
    return alpha * key + (1-alpha) * img

@torch.no_grad()
def test(model, dataloader, triggers, args):
    model.eval()
    cnt, ACC = 0, 0
    trapdoor_ACC = 0

    if triggers is not None:
        alpha = args['trapdoor']['alpha']
    n_classes = args['dataset']['n_classes']

    dataset_name = args['dataset']['name']

    data_size = dataloader.batch_size * len(dataloader)
    num_trapdoor, last_trapdoor = divmod(data_size, n_classes)

    trapdoor_iden_iterator = torch.hstack([
        torch.arange(0, n_classes).repeat(num_trapdoor),
        torch.randperm(n_classes)[:last_trapdoor]
    ])[torch.randperm(data_size)].split(dataloader.batch_size)

    for (img, iden), trapdoor_iden in zip(dataloader, trapdoor_iden_iterator):
        img, iden = img.cuda(), iden.cuda()
        bs = img.size(0)
        iden = iden.view(-1)

        out_prob = model(img)[-1]
        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

        if triggers is not None:
            trapdoor_iden = trapdoor_iden.cuda()
            key = torch.stack([triggers[j] for j in trapdoor_iden], dim=0)
            if dataset_name == 'mnist':
                key = key.expand(-1, 3, -1, -1)
            trapdoor_img = blend(img, key, alpha)

            trapdoor_out_prob = model(trapdoor_img)[-1]
            trapdoor_out_iden = torch.argmax(trapdoor_out_prob, dim=1).view(-1)
            trapdoor_ACC += torch.sum(trapdoor_iden == trapdoor_out_iden).item()

    return ACC * 100.0 / cnt, trapdoor_ACC * 100.0 / cnt

def train(args, model, criterion, optimizer, trainloader, testloader, n_epochs, triggers=None, scheduler=None, trapdoor_criterion=None):
    root_path = args['root_path']
    dataset_name = args['dataset']['name']

    best_ACC = 0.0
    final_trapdoor_ACC = 0
    model_name = args['dataset']['model_name']

    n_classes = args['dataset']['n_classes']
    data_size = trainloader.batch_size * len(trainloader)
    num_trapdoor, last_trapdoor = divmod(data_size, n_classes)

    if triggers is not None:
        # Only use augmentation when incorporating trapdoors
        aug_list = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0), p=0.5),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.RandomRotation(30, p=0.5),
        )

        alpha = args['trapdoor']['alpha']
        beta = args['trapdoor']['beta']

        final_triggers = deepcopy(triggers)

        triggers = triggers.cuda()
        if args['trapdoor']['optimized']:
            trigger_step = args['trapdoor']['step_size']
            if args['trapdoor']['discriminator_loss']:
                pretrained_path = args['pretrained_path']
                D = utils.init_model(model_name, 1, pretrained_path)
                optimizer_D, scheduler_D = utils.init_optimizer(args[model_name], D.parameters())
                D = nn.DataParallel(D).cuda()

            if args['trapdoor']['discriminator_feat_loss']:
                D_feat = nn.Sequential(
                    nn.Linear(model.module.feat_dim + n_classes, model.module.feat_dim + n_classes),
                    nn.ReLU(),
                    nn.Linear(model.module.feat_dim + n_classes, 1)
                )
                optimizer_D_feat, scheduler_D_feat = utils.init_optimizer(args[model_name], D_feat.parameters())
                D_feat = nn.DataParallel(D_feat).cuda()

        if trapdoor_criterion is None:
            trapdoor_criterion = deepcopy(criterion)
    else:
        aug_list = lambda x: x
        final_triggers = None

    for epoch in range(n_epochs):
        tf = time.time()
        cnt = 0
        ACC, loss_tot = 0, 0
        main_loss_tot = 0
        trapdoor_ACC, trapdoor_loss_tot = 0, 0

        # LS scheduler
        if callable(getattr(criterion, 'step', None)):
            criterion.step(epoch, n_epochs)
        if triggers is not None and callable(getattr(trapdoor_criterion, 'step', None)):
            trapdoor_criterion.step(epoch, n_epochs)

        trapdoor_iden_iterator = torch.hstack([
            torch.arange(0, n_classes).repeat(num_trapdoor),
            torch.randperm(n_classes)[:last_trapdoor]
        ])[torch.randperm(data_size)].split(trainloader.batch_size)

        for i, ((img, iden), trapdoor_iden) in enumerate(zip(trainloader, trapdoor_iden_iterator)):
            img, iden = img.cuda(), iden.cuda()
            trapdoor_iden = trapdoor_iden.cuda()
            bs = img.size(0)
            iden = iden.view(-1)

            """
            Update discriminator and triggers
            """
            if triggers is not None and args['trapdoor']['optimized']:
                model.eval()
                triggers.requires_grad = True

                key = torch.stack([triggers[j] for j in trapdoor_iden], dim=0)
                if dataset_name == 'mnist':
                    key = key.expand(-1, 3, -1, -1)
                trapdoor_img = blend(img, key, alpha)

                trigger_loss = 0
                if args['trapdoor']['discriminator_loss']:
                    # Train discriminator
                    D.train()
                    concat_feat, concat_prob = D(torch.concat([img, trapdoor_img.detach()]))
                    D_loss = nn.BCEWithLogitsLoss()(
                        concat_prob,
                        torch.concat([torch.ones((bs, 1)), torch.zeros((bs, 1))]).cuda()
                    )
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()

                    # Train triggers
                    D.eval()
                    trapdoor_feats, trapdoor_out_prob = D(trapdoor_img)
                    trigger_loss += nn.BCEWithLogitsLoss()(
                        trapdoor_out_prob,
                        torch.ones((bs, 1)).cuda()
                    ) * (0.5 if args['trapdoor']['discriminator_feat_loss'] else 1)

                if args['trapdoor']['discriminator_feat_loss']:
                    # Train feature-level discriminator
                    D_feat.train()
                    concat_feat, concat_out_prob = model(torch.concat([img, trapdoor_img.detach()]))
                    concat_feat = torch.hstack([concat_feat, (concat_out_prob == concat_out_prob.max(dim=1).values.unsqueeze(dim=1)).float()]).detach()
                    concat_prob = D_feat(concat_feat)
                    D_feat_loss = nn.BCEWithLogitsLoss()(
                        concat_prob,
                        torch.concat([torch.ones((bs, 1)), torch.zeros((bs, 1))]).cuda()
                    )
                    optimizer_D_feat.zero_grad()
                    D_feat_loss.backward()
                    optimizer_D_feat.step()

                    # Train triggers
                    D_feat.eval()
                    trapdoor_feats, trapdoor_out_prob = model(trapdoor_img)
                    trapdoor_feats = torch.hstack([trapdoor_feats, (trapdoor_out_prob == trapdoor_out_prob.max(dim=1).values.unsqueeze(dim=1)).float().detach()])
                    trapdoor_out_prob = D_feat(trapdoor_feats)
                    trigger_loss += nn.BCEWithLogitsLoss()(
                        trapdoor_out_prob,
                        torch.ones((bs, 1)).cuda()
                    ) * (0.5 if args['trapdoor']['discriminator_loss'] else 1)

                aug_trapdoor_img = aug_list(trapdoor_img)
                trigger_feats, trigger_out_prob = model(aug_trapdoor_img)
                trigger_loss += trapdoor_criterion(trigger_out_prob, trapdoor_iden)

                trigger_loss.backward()
                grad = triggers.grad.data
                triggers.data = (triggers.data - trigger_step * grad.sign()).clamp(min=0, max=1)
                triggers.grad.detach_()
                triggers.grad.zero_()

                triggers.requires_grad = False

            """
            Update model
            """
            model.train()

            aug_img = aug_list(img)

            if triggers is not None:
                key = torch.stack([triggers[j] for j in trapdoor_iden], dim=0)
                if dataset_name == 'mnist':
                    key = key.expand(-1, 3, -1, -1)
                trapdoor_img = blend(img, key, alpha)
                aug_trapdoor_img = aug_list(trapdoor_img)

                concat_feat, concat_prob = model(torch.concat([aug_img, aug_trapdoor_img]))
                feats, out_prob = concat_feat[:bs], concat_prob[:bs]
                trapdoor_feats, trapdoor_out_prob = concat_feat[bs:], concat_prob[bs:]

                cross_loss = criterion(out_prob, iden)

                trapdoor_loss = trapdoor_criterion(trapdoor_out_prob, trapdoor_iden)

                discriminator_loss = 0
                if args['trapdoor']['discriminator_feat_model_loss']:
                    D_feat.eval()
                    concat_feat = torch.hstack([concat_feat, (concat_prob == concat_prob.max(dim=1).values.unsqueeze(dim=1)).float().detach()])
                    concat_dis_prob = D_feat(concat_feat)
                    discriminator_loss = nn.BCEWithLogitsLoss()(
                        concat_dis_prob,
                        torch.concat([0.5 * torch.ones((bs, 1)), 0.5 * torch.ones((bs, 1))]).cuda()
                    )

                loss = (1-beta) * cross_loss + beta * trapdoor_loss + beta * discriminator_loss
            else:
                feats, out_prob = model(aug_img)
                cross_loss = criterion(out_prob, iden)
                loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

            main_loss_tot += cross_loss.item() * bs
            if triggers is not None:
                trapdoor_loss_tot += trapdoor_loss.item() * bs
                trapdoor_out_iden = torch.argmax(trapdoor_out_prob, dim=1).view(-1)
                trapdoor_ACC += torch.sum(trapdoor_iden == trapdoor_out_iden).item()

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        train_main_loss = main_loss_tot * 1.0 / cnt
        train_trapdoor_loss = trapdoor_loss_tot * 1.0 / cnt
        train_trapdoor_acc = trapdoor_ACC * 100.0 / cnt

        test_acc, test_trapdoor_acc = test(model, testloader, triggers, args)

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)
            if triggers is not None:
                final_trapdoor_ACC = test_trapdoor_acc
                final_triggers = deepcopy(triggers)

        if (epoch+1) % 10 == 0:
            model_path = os.path.join(root_path, "target_ckp")
            torch.save({ 'state_dict': model.state_dict() }, os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))
            if triggers is not None and args['trapdoor']['optimized']:
                trigger_path = args['trigger_path']
                torch.save(triggers, os.path.join(trigger_path, "trigger_epoch{}.tar").format(epoch))

        print("Epoch:{} | Time:{:.2f} | Train Loss:{:.2f} | Train Main Loss:{:.2f} | Train trapdoor Loss:{:.2f} | Train Acc:{:.2f} | Train trapdoor Acc:{:.2f} | Test Acc:{:.2f} | Test trapdoor Acc:{:.2f}".format(
             epoch, interval, train_loss, train_main_loss, train_trapdoor_loss, train_acc, train_trapdoor_acc, test_acc, test_trapdoor_acc
        ))
        if scheduler is not None:
            scheduler.step()
        if triggers is not None:
            if args['trapdoor']['discriminator_loss'] and scheduler_D is not None:
                scheduler_D.step()
            if args['trapdoor']['discriminator_feat_loss'] and scheduler_D_feat is not None:
                scheduler_D_feat.step()

    if triggers is not None and args['trapdoor']['discriminator_loss']:
        torch.save({ 'state_dict': D.state_dict() }, os.path.join(root_path, "discriminator.tar"))

    if triggers is not None and args['trapdoor']['discriminator_feat_loss']:
        torch.save({ 'state_dict': D_feat.state_dict() }, os.path.join(root_path, "discriminator_feat.tar"))

    print("Best Acc:{:.2f} | trapdoor Acc:{:.2f}".format(best_ACC, final_trapdoor_ACC))
    return best_model, best_ACC, final_trapdoor_ACC, final_triggers