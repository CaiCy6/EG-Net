import os

import numpy as np
import argparse
from datetime import datetime
import logging
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore")
from models.EG-Net import ConvNeXtEGNet as model


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, args.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)  # forward

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)  # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze()  # apply sigmoid activation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # eval Dice
        input = res >= 0.5
        target = np.array(gt >= 0.5)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path, model_name='SUFormer'):
    model.train()
    global best
    global total_train_time
    global max_memory_usage  # 添加一个全局变量来存储最大显存占用
    max_memory_usage = 0  # 初始化最大显存占用
    time_before_epoch_start = time.time()
    size_rates = [1]
    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(args.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            outputs = model(images)
            loss = structure_loss(outputs, gts)

            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, args.batchsize)

            current_memory = torch.cuda.max_memory_allocated()
            max_memory_usage = max(max_memory_usage, current_memory)

        if i % 50 == 0 or i == total_step:
            inter_info = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, max memory usage: {:.2f} MB'.format(
                datetime.now(), epoch, args.epoch, i, total_step, loss_record.show(), max_memory_usage / (1024 ** 2)
            )
            print(inter_info)
            logging.info(inter_info)

    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    logging.info('total train time till current epoch: ' + str(total_train_time))

    save_path = (args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')

    global dict_plot

    if (epoch) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['test']:
            dataset_dice, n_images = test(model, test_path, dataset)
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

        dataset_test_dice = total_dice / total_images
        meandice = dataset_test_dice
        valid_meandice = total_dice / total_images
        dict_plot['valid'].append(valid_meandice)
        dict_plot['test'].append(dataset_test_dice)
        print('Test dice score: {}'.format(dataset_test_dice))
        logging.info('Test dice score: {}'.format(dataset_test_dice))

        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '-best.pth')

    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    from models.UNet import UNet2D as model
    
    warnings.filterwarnings("ignore")
    dict_plot = {'valid': [], 'test': []}
    name = ['valid', 'test']
    model_name = 'EG-Net'
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, default=1, help="output channel of network")
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")

    parser.add_argument('--img_size', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='/root/autodl-tmp/datasets/HAM10000/train/',
                        help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='/root/autodl-tmp/datasets/HAM10000',
                        help='path to testing Kvasir dataset')
    parser.add_argument('--train_save', type=str, default='./model_out/HAM10000/' + model_name + '/')

    args = parser.parse_args()

    if not os.path.exists(args.train_save):
        os.makedirs(args.train_save)

    logging.basicConfig(filename='./model_out/HAM10000/' + model_name + '/train.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')


    pretrained_weights_path = '/root/autodl-tmp/MSDUNet/premodelweight/pytorch_model_384_small_new.bin'
    model = model().cuda(0)
    
    
    logging.info(model)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    macs, params = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    best = 0

    params = model.parameters()

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.img_size, shuffle=True,
                              augmentation=args.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        train(train_loader, model, optimizer, epoch, args.test_path, model_name=model_name)
    print('avg train time: ' + str(total_train_time / (args.epoch - 1)))
    logging.info('avg train time: ' + str(total_train_time / (args.epoch - 1)))
