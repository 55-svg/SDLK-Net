import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from Encoder_Prototype_rgbd_endT import Mnet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from smooth_loss import get_saliency_smoothness
import torch.backends.cudnn as cudnn
from options import opt
import logging

def ensure_dir_exists(directory):
    """Ensure that the directory exists. Create it if it does not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure the save path directory exists
ensure_dir_exists(opt.save_path)

# Set up logging
log_file = os.path.join(opt.save_path, 'training.log')

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# Set the device for training
cudnn.benchmark = True

train_image_root = opt.train_rgb_root
train_gt_root = opt.train_gt_root
train_t_root = opt.train_t_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_t_root = opt.val_t_root

save_path = opt.save_path
model = Mnet()
num_parms = 0
model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logger.info(f"Total Parameters (For Reference): {num_parms}")

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# Load data
logger.info('Loading data...')
train_loader = get_loader(train_image_root, train_gt_root, train_t_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, val_t_root, testsize=384)
total_step = len(train_loader)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

# Set loss function
step = 0
best_mae = 1
best_mae_epoch = 0
best_avg_loss = 1
best_avg_epoch = 0
dice = SoftDiceLoss()

def train(train_loader, model, optimizer, epoch, save_path):
    global step, best_avg_loss, best_avg_epoch
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, t) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            rgb = images.cuda()
            gts = gts.cuda()
            t = t.cuda()

            out1_fusion, out2_fusion, out3_fusion, out4_fusion = model(rgb, t)
            sml = get_saliency_smoothness(torch.sigmoid(out1_fusion), gts)
            loss1_fusion = F.binary_cross_entropy_with_logits(out1_fusion, gts)
            loss2_fusion = F.binary_cross_entropy_with_logits(out2_fusion, gts)
            loss3_fusion = F.binary_cross_entropy_with_logits(out3_fusion, gts)
            loss4_fusion = F.binary_cross_entropy_with_logits(out4_fusion, gts)
            dice_loss = dice(out1_fusion, gts)
            loss_seg = loss1_fusion + loss2_fusion * 0.8 + loss3_fusion * 0.6 + loss4_fusion * 0.5 + sml + dice_loss

            loss = loss_seg
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.item()
            if i % 30 == 0 or i == total_step or i == 1:
                logger.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], LR:{optimizer.state_dict()["param_groups"][0]["lr"]:.7f} || sal_loss:{loss.item():.4f}')

        loss_all /= epoch_step
        logger.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}] || epoch_avg_loss:{loss_all:.4f} ... best avg loss: {best_avg_loss:.4f} ... best avg epoch: {best_avg_epoch}')
        if epoch == 1:
            best_avg_loss = loss_all
        else:
            if best_avg_loss >= loss_all:
                best_avg_loss = loss_all
                best_avg_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, '3090_test_SPNet_Best_AVG_epoch_20p.pth'))
                logger.info(f'Best avg epoch: {epoch}')
        if epoch == 200:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_3090_test_SPNet_20p.pth'))

    except KeyboardInterrupt:
        logger.error('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger.info('Save checkpoints successfully!')
        raise

def val(test_loader, model, epoch, save_path):
    global best_mae, best_mae_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            score, score1, score2, s_sig = model(image, depth)
            res = F.interpolate(score, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        logger.info(f'SPNet_Epoch: {epoch} - MAE: {mae:.4f} ... best mae: {best_mae:.4f} ... best mae Epoch: {best_mae_epoch}')
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_mae_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, '3090_test_SPNet_Best_MAE_epoch_20p.pth'))
                logger.info(f'SPNet mae epoch: {epoch}')

if __name__ == '__main__':
    logger.info("SPNet_Start train...")
    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch % 2 == 0:
            val(test_loader, model, epoch, save_path)
