from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from model import ESSA_UNet

from dataset import fusion_dataset_gt
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from loss import fusion_loss_adv
import torch
from torch.utils.data import DataLoader
import warnings
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from torchvision.transforms import Resize

from collections import OrderedDict

from torch.nn.parallel import DataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings('ignore')

def seed_everything(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def attack(image_vis, image_ir, image_gt, model, loss, step_size=1/255, total_steps=3, epsilon=4/255):
    # seed_everything()
    model.eval()
    image_vis = image_vis.cuda()
    image_ir = image_ir.cuda()
    image_gt = image_gt.cuda()
    image_gt_ycbcr = RGB2YCrCb(image_gt)
    image_gt_ycbcr = image_gt_ycbcr[:, : 1]
    image_vis_ycrcb = RGB2YCrCb(image_vis)


    random_init_vis = torch.rand_like(image_vis)*0.0003
    random_init_ir = torch.rand_like(image_ir)*0.0003
    adv_img_vis = Variable(torch.clamp(image_vis.data+random_init_vis.data,0,1), requires_grad = True)
    adv_img_ir = Variable(torch.clamp(image_ir.data+random_init_ir.data,0,1), requires_grad = True)

    for t in range(total_steps):

        adv_image_vis_ycrcb = RGB2YCrCb(adv_img_vis)
        logits = model(adv_image_vis_ycrcb, adv_img_ir)
        # loss_total, loss_in, loss_grad, loss_ssim =  loss(image_vis_ycrcb, image_ir, logits)
        loss_total, loss_mse, loss_ssim = loss(logits, image_gt_ycbcr)
        loss_total.backward()
        print(loss_total)

        with torch.no_grad():
            # update image_vis_adv
            grad_info = step_size * adv_img_vis.grad.data.sign()
            adv_img_vis = adv_img_vis.data + grad_info
            eta = torch.clamp(adv_img_vis.data - image_vis.data, -epsilon, epsilon)
            adv_img_vis = image_vis.data + eta

            grad_info = step_size * adv_img_ir.grad.data.sign()
            adv_img_ir = adv_img_ir.data + grad_info
            eta = torch.clamp(adv_img_ir.data - image_ir.data, -epsilon, epsilon)
            adv_img_ir = image_ir.data + eta
        
        adv_img_vis = Variable(torch.clamp(adv_img_vis.data,0,1), requires_grad = True)
        adv_img_ir = Variable(torch.clamp(adv_img_ir.data,0,1), requires_grad = True)
    
    return adv_img_vis, adv_img_ir



def train(logger=None):
    seed_everything()
    lr_start = 0.001
    model_path = './model'
    model_path = os.path.join(model_path)
    
    train_model = ESSA_UNet()
    train_model.cuda()
    train_model.train()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr_start)

    ir_path = '/home/lijiawei/local/dataset/train/ir/'
    vis_path = '/home/lijiawei/local/dataset/train/vis/'
    gt_path = '/home/lijiawei/local/dataset/train/gt/'


    train_dataset = fusion_dataset_gt(ir_path=ir_path, vis_path=vis_path, gt_path=gt_path)
    print("The length of training dataset:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    train_loss = fusion_loss_adv()


    epoch = 50

    st = glob_st = time.time()
    logger.info('Train start!')

    # plot_loss_total = []  # 画图用

    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epo
        for it, (image_vis, image_ir, image_gt, name) in enumerate(train_loader):
            train_model.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            image_gt = Variable(image_gt).cuda()
            image_gt_ycbcr = RGB2YCrCb(image_gt)
            image_gt_ycbcr = image_gt_ycbcr[:, : 1]

            logits = train_model(image_vis_ycrcb, image_ir)  # inputs
            
            # 生成对抗样本
            # image_vis_adv, image_ir_adv = attack_think2(image_vis, image_ir, logits, train_model, train_loss_adv)
            image_vis_adv, image_ir_adv = attack(image_vis, image_ir, image_gt, train_model, train_loss)

            logits_adv = train_model(image_vis_adv, image_ir_adv)
            

            optimizer.zero_grad()


            loss_total, loss_mse, loss_ssim = train_loss(logits, image_gt_ycbcr)
            loss_total_adv, loss_mse_adv, loss_ssim_adv = train_loss(logits_adv, image_gt_ycbcr)

            loss = loss_total + loss_total_adv
            loss.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss: {loss:.4f}',
                        'loss_total: {loss_total:.4f}',
                        'loss_mse: {loss_mse:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',

                        # 对抗loss
                        'loss_total_adv: {loss_total_adv:.4f}',
                        'loss_mse_adv: {loss_mse_adv:.4f}',
                        'loss_ssim_adv: {loss_ssim_adv:.4f}',

                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        loss=loss.item(),
                        loss_total=loss_total.item(),
                        loss_mse=loss_mse.item(),
                        loss_ssim=loss_ssim.item(),

                        # 对抗loss
                        loss_total_adv=loss_total_adv.item(),
                        loss_mse_adv=loss_mse_adv.item(),
                        loss_ssim_adv=loss_ssim_adv.item(),

                        time=t_intv,
                        eta=eta,
                    )
                logger.info(msg)
                st = ed

                # loss图
                # plot_loss_total.append((loss_total.cpu().detach().numpy()))
                # loss_filename_path1 = "D:\\Python\\code_lok18\\myMedicalCode\\loss_chat\\" + "loss-total_a=0.8" + ".mat"
                # scio.savemat(loss_filename_path1, {'loss_total': plot_loss_total})

    train_model_file = os.path.join(model_path, 'rebuttal_13.pth')
    torch.save(train_model.state_dict(), train_model_file)
    logger.info("Train model save as: {}".format(train_model_file))
    logger.info('\n')

if __name__ == "__main__":
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    train(logger)
    print("Train finish!")





