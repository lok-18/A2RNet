from PIL import Image
import numpy as np
import sys
from torch.autograd import Variable

from model import ESSA_UNet

from dataset import fusion_dataset_gt
# from TaskFusion_dataset import Fusion_dataset_attack as fusion_dataset
import argparse
import datetime
import time
import logging

import os
from logger import setup_logger
from loss import fusion_loss_adv
import torch
from torch.utils.data import DataLoader
import warnings
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
from torchvision.transforms import Resize
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random
from collections import OrderedDict

from torchvision.transforms import Resize

from torch.nn.parallel import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings('ignore')



def main(args):
    lr_start = 0.001
    model_path = './model'
    model_path = os.path.join(model_path)
    fusion_model_path = os.path.join(model_path, args.checkpoint)
    
    if args.modelname == "backbone":
        
        train_model = ESSA_UNet()
        # train_model = backbone()
        # train_model = UNet()

        # train_model = DataParallel(train_model)
    elif args.modelname == "SortHybridModelFusion":
        # train_model = SortHybridModelFusion()
        train_model = DataParallel(train_model)
    else:
        assert False,"No Model is inited."
    # train_model = SortHybridModelFusion()
    train_model.cuda()

    train_model.load_state_dict(torch.load(fusion_model_path))
    print('Model loading...')


    img_path = args.indir
 
    ir_path = img_path + '/ir/'
    vis_path = img_path + '/vis/'
    gt_path = "./test_images/gt"


    test_dataset = fusion_dataset_gt(ir_path=ir_path, vis_path=vis_path, gt_path=gt_path)
    print("The length of attacking dataset:{}".format(test_dataset.length))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_loader.n_iter = len(test_loader)

    train_loss = fusion_loss_adv()

    st = glob_st = time.time()
    print('Attack start!')


    for it, (image_vis, image_ir, image_gt, name) in enumerate(test_loader):
        adv_image_vis, adv_image_ir, loss_total= attack(image_vis, image_ir, image_gt, train_model, train_loss, step_size = args.step_size, total_steps = args.total_steps, epsilon=args.epsilon)
        with torch.no_grad():
            adv_image_vis_ycrcb = RGB2YCrCb(adv_image_vis)
            logits = train_model(adv_image_vis_ycrcb, adv_image_ir)

            fusion_ycrcb = torch.cat(
                (logits, adv_image_vis_ycrcb[:, 1:2, :, :], adv_image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)

            adv_vis = adv_image_vis.cpu().numpy()
            adv_vis = adv_vis.transpose((0, 2, 3, 1))
            adv_vis = (adv_vis - np.min(adv_vis)) / (
                    np.max(adv_vis) - np.min(adv_vis)
            )
            adv_vis = np.uint8(255.0 * adv_vis)

            adv_ir = adv_image_ir.cpu().numpy()
            adv_ir = adv_ir.transpose((0, 2, 3, 1))
            adv_ir = (adv_ir - np.min(adv_ir)) / (
                    np.max(adv_ir) - np.min(adv_ir)
            )
            adv_ir = np.uint8(255.0 * adv_ir)

            st = time.time()
            for k in range(len(name)):
                image_fused = fused_image[k, :, :, :]
                adv_vis = adv_vis[k, :, :, :]
                adv_ir = adv_ir[k, :, :, 0]

                image_fused = Image.fromarray(image_fused)
                adv_vis = Image.fromarray(adv_vis)
                adv_ir = Image.fromarray(adv_ir)

                save_path_fused = os.path.join(args.outdir+'/fused', name[k])
                save_path_vis =  os.path.join(args.outdir+'/advvis', name[k])
                save_path_ir = os.path.join(args.outdir+'/advir', name[k])

                image_fused.save(save_path_fused)
                adv_vis.save(save_path_vis)
                adv_ir.save(save_path_ir)
                
                ed = time.time()
                print('file_name: {0}'.format(save_path_fused))
                print('Time:', ed - st)



def attack(image_vis, image_ir, image_gt, model, loss, step_size, total_steps, epsilon):
    model.eval()
    image_vis = image_vis.cuda()
    image_ir = image_ir.cuda()
    image_gt = image_gt.cuda()
    image_gt_ycbcr = RGB2YCrCb(image_gt)
    image_gt_ycbcr = image_gt_ycbcr[:, : 1]
    image_vis_ycrcb = RGB2YCrCb(image_vis)

    torch_resize = Resize([480,640])
    image_vis_resized = torch_resize(image_vis_ycrcb)
    image_ir_resized = torch_resize(image_ir)
    image_gt_resized = torch_resize(image_gt_ycbcr)

    random_init_vis = torch.rand_like(image_vis)*0.0003
    random_init_ir = torch.rand_like(image_ir)*0.0003
    adv_img_vis = Variable(torch.clamp(image_vis.data+random_init_vis.data,0,1), requires_grad = True)
    adv_img_ir = Variable(torch.clamp(image_ir.data+random_init_ir.data,0,1), requires_grad = True)

    for t in range(total_steps):

        adv_img_ir_ = torch_resize(adv_img_ir)
        adv_img_vis_ = torch_resize(adv_img_vis)

        adv_image_vis_ycrcb = RGB2YCrCb(adv_img_vis_)
        logits = model(adv_image_vis_ycrcb, adv_img_ir_)
        # loss_total, loss_in, loss_grad, loss_ssim =  loss(image_vis_ycrcb, image_ir, logits)
        loss_total, loss_mse, loss_ssim = loss(logits, image_gt_resized)
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
    
    return adv_img_vis, adv_img_ir, loss_total




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="./test_images",  # MSRS
        nargs="?",
        help="dir containing image pairs",
    )
    parser.add_argument(
        "--outdir",
        default="./results",
        type=str,
        nargs="?",
        help="dir containing image pairs",
    )
    parser.add_argument(
        "--modelname",
        default="backbone",
        type=str,
        nargs="?",
        help="dir containing image pairs",
    )
    parser.add_argument(
        "--checkpoint",
        default="model.pth",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="PGD",
        help="Attacking methods,[AdvDM, MFA, Embedding attack, clean]",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default="0.003",
        help="(1/255)The step size when optimizing adversarial examples",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default="20",
        help="The iterations of optimizings",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default="0.015",
        help="(4/255)The l∞ limits when optimizing adversarial examples",
    )
    
    args = parser.parse_args()
    os.makedirs(args.outdir+'/fused', mode=0o777, exist_ok=True)
    os.makedirs(args.outdir+'/advvis', mode=0o777, exist_ok=True)
    os.makedirs(args.outdir+'/advir', mode=0o777, exist_ok=True)
    main(args)
    print("Attack finish!")


