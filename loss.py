import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

    
class fusion_loss_adv(nn.Module):
    def __init__(self):
        super(fusion_loss_adv, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, generate_img_adv, generate_img):
        # L_mse
        loss_mse = F.mse_loss(generate_img_adv, generate_img)

        # L_ssim
        loss_ssim = 1 - ssim(generate_img_adv, generate_img)

        # L_total
        loss_total = 100 * loss_mse + 100 * loss_ssim
        return loss_total, loss_mse, loss_ssim



if __name__ == '__main__':
    pass

