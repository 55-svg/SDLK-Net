# import os
# from PIL import Image
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import random
# import numpy as np
# from PIL import ImageEnhance
# from skimage.filters import threshold_multiotsu
#
#
# # several data augumentation strategies
# def cv_random_flip(img, label, depth):
#     flip_flag = random.randint(0, 1)
#     # flip_flag2= random.randint(0,1)
#     # left right flip
#     if flip_flag == 1:
#         img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)
#         depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
#     # top bottom flip
#     # if flip_flag2==1:
#     #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
#     #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
#     #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
#     return img, label, depth
#
#
# def randomCrop(image, label, depth):
#     border = 30
#     image_width = image.size[0]
#     image_height = image.size[1]
#     crop_win_width = np.random.randint(image_width - border, image_width)
#     crop_win_height = np.random.randint(image_height - border, image_height)
#     random_region = (
#         (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
#         (image_height + crop_win_height) >> 1)
#     return image.crop(random_region), label.crop(random_region), depth.crop(random_region)
#
#
# def randomRotation(image, label, depth):
#     mode = Image.BICUBIC
#     if random.random() > 0.8:
#         random_angle = np.random.randint(-15, 15)
#         image = image.rotate(random_angle, mode)
#         label = label.rotate(random_angle, mode)
#         depth = depth.rotate(random_angle, mode)
#     return image, label, depth
#
#
# def colorEnhance(image):
#     bright_intensity = random.randint(5, 15) / 10.0
#     image = ImageEnhance.Brightness(image).enhance(bright_intensity)
#     contrast_intensity = random.randint(5, 15) / 10.0
#     image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
#     color_intensity = random.randint(0, 20) / 10.0
#     image = ImageEnhance.Color(image).enhance(color_intensity)
#     sharp_intensity = random.randint(0, 30) / 10.0
#     image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
#     return image
#
#
# def randomGaussian(image, mean=0.1, sigma=0.35):
#     def gaussianNoisy(im, mean=mean, sigma=sigma):
#         for _i in range(len(im)):
#             im[_i] += random.gauss(mean, sigma)
#         return im
#
#     img = np.asarray(image)
#     width, height = img.shape
#     img = gaussianNoisy(img[:].flatten(), mean, sigma)
#     img = img.reshape([width, height])
#     return Image.fromarray(np.uint8(img))
#
#
# def randomPeper(img):
#     img = np.array(img)
#     noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
#     for i in range(noiseNum):
#
#         randX = random.randint(0, img.shape[0] - 1)
#
#         randY = random.randint(0, img.shape[1] - 1)
#
#         if random.randint(0, 1) == 0:
#
#             img[randX, randY] = 0
#
#         else:
#
#             img[randX, randY] = 255
#     return Image.fromarray(img)
#
#
# # dataset for training
# # The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# # (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
#
# class SalObjDataset(data.Dataset):
#     def __init__(self, image_root, gt_root, depth_root, trainsize):
#         self.trainsize = trainsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
#                     or f.endswith('.png')]
#         self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
#                        or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.depths = sorted(self.depths)
#         self.filter_files()
#         self.size = len(self.images)
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.outputIASSF, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         self.depths_transform = transforms.Compose(
#             [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
#
#     def __getitem__(self, index):
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
#         depth = self.binary_loader(self.depths[index])
#         image, gt, depth = cv_random_flip(image, gt, depth)
#         image, gt, depth = randomCrop(image, gt, depth)
#         image, gt, depth = randomRotation(image, gt, depth)
#         image = colorEnhance(image)
#         # gt=randomGaussian(gt)
#         gt = randomPeper(gt)
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         depth = self.depths_transform(depth)
#         d = depth.numpy()
#         # print(d.shape)
#         regions = np.digitize(d[0], bins=threshold_multiotsu(d[0]))
#         h, w = d[0].shape
#         nb = 3
#         bin = np.zeros((nb, h, w))
#         for i in range(nb):
#             bin[i] = regions == i
#         # print(depth.type())
#         # print(torch.from_numpy(bin).float().type())
#         return image, gt, depth, torch.from_numpy(bin).float()
#
#     def filter_files(self):
#         assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
#         images = []
#         gts = []
#         depths = []
#         for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             depth = Image.open(depth_path)
#             if img.size == gt.size and gt.size == depth.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 depths.append(depth_path)
#         self.images = images
#         self.gts = gts
#         self.depths = depths
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def resize(self, img, gt, depth):
#         assert img.size == gt.size and gt.size == depth.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
#                                                                                                       Image.NEAREST)
#         else:
#             return img, gt, depth
#
#     def __len__(self):
#         return self.size
#
#
# ###############################################################################
# # 0919
# #
#
# class SalObjDataset_var(data.Dataset):
#     def __init__(self, image_root, gt_root, depth_root, trainsize):
#
#         self.trainsize = trainsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.depths = sorted(self.depths)
#         self.filter_files()
#         self.size = len(self.images)
#
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.outputIASSF, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         self.depths_transform = transforms.Compose(
#             [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
#
#     def __getitem__(self, index):
#
#         ## read imag, gt, depth
#         image0 = self.rgb_loader(self.images[index])
#         gt0 = self.binary_loader(self.gts[index])
#         depth0 = self.binary_loader(self.depths[index])
#
#         ##################################################
#         ## out1
#         ##################################################
#         image, gt, depth = cv_random_flip(image0, gt0, depth0)
#         image, gt, depth = randomCrop(image, gt, depth)
#         image, gt, depth = randomRotation(image, gt, depth)
#         image = colorEnhance(image)
#         gt = randomPeper(gt)
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         depth = self.depths_transform(depth)
#
#         ##################################################
#         ## out1
#         ##################################################
#         image2, gt2, depth2 = cv_random_flip(image0, gt0, depth0)
#         image2, gt2, depth2 = randomCrop(image2, gt2, depth2)
#         image2, gt2, depth2 = randomRotation(image2, gt2, depth2)
#         image2 = colorEnhance(image2)
#         gt2 = randomPeper(gt2)
#         image2 = self.img_transform(image2)
#         gt2 = self.gt_transform(gt2)
#         depth2 = self.depths_transform(depth2)
#
#         return image, gt, depth, image2, gt2, depth2
#
#     def filter_files(self):
#
#         assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
#         images = []
#         gts = []
#         depths = []
#         for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             depth = Image.open(depth_path)
#             if img.size == gt.size and gt.size == depth.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 depths.append(depth_path)
#         self.images = images
#         self.gts = gts
#         self.depths = depths
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def resize(self, img, gt, depth):
#         assert img.size == gt.size and gt.size == depth.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
#                                                                                                       Image.NEAREST)
#         else:
#             return img, gt, depth
#
#     def __len__(self):
#         return self.size
#
#
# class SalObjDataset_var_unlabel(data.Dataset):
#     def __init__(self, image_root, gt_root, depth_root, trainsize):
#
#         self.trainsize = trainsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.depths = sorted(self.depths)
#         self.filter_files()
#         self.size = len(self.images)
#
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.outputIASSF, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         self.depths_transform = transforms.Compose(
#             [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
#
#     def __getitem__(self, index):
#
#         ## read imag, gt, depth
#         image0 = self.rgb_loader(self.images[index])
#         gt0 = self.binary_loader(self.gts[index])
#         depth0 = self.binary_loader(self.depths[index])
#
#         ##################################################
#         ## out1
#         ##################################################
#         image, gt, depth = cv_random_flip(image0, gt0, depth0)
#         image, gt, depth = randomCrop(image, gt, depth)
#         image, gt, depth = randomRotation(image, gt, depth)
#         image = colorEnhance(image)
#         gt = randomPeper(gt)
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         depth = self.depths_transform(depth)
#
#         ##################################################
#         ## out1
#         ##################################################
#         image2, gt2, depth2 = cv_random_flip(image0, gt0, depth0)
#         image2, gt2, depth2 = randomCrop(image2, gt2, depth2)
#         image2, gt2, depth2 = randomRotation(image2, gt2, depth2)
#         image2 = colorEnhance(image2)
#         gt2 = randomPeper(gt2)
#         image2 = self.img_transform(image2)
#         gt2 = self.gt_transform(gt2)
#         depth2 = self.depths_transform(depth2)
#
#         return image, gt, depth, image2, gt2, depth2
#
#     def filter_files(self):
#
#         assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
#         images = []
#         gts = []
#         depths = []
#         for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             depth = Image.open(depth_path)
#             if img.size == gt.size and gt.size == depth.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 depths.append(depth_path)
#         self.images = images
#         self.gts = gts
#         self.depths = depths
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def resize(self, img, gt, depth):
#         assert img.size == gt.size and gt.size == depth.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
#                                                                                                       Image.NEAREST)
#         else:
#             return img, gt, depth
#
#     def __len__(self):
#         return self.size
#
#
# # dataloader for training
# def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
#     dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader
#
#
# # dataloader for training2
# ## 09-19-2020
# def get_loader_var(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12,
#                    pin_memory=False):
#     dataset = SalObjDataset_var(image_root, gt_root, depth_root, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader
#
#
# def get_loader_var_unlabel(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12,
#                            pin_memory=False):
#     dataset = SalObjDataset_var_unlabel(image_root, gt_root, depth_root, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader
#
#
# # test dataset and loader
# class test_dataset:
#     def __init__(self, image_root, gt_root, depth_root, testsize):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
#                     or f.endswith('.png')]
#         self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
#                        or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.depths = sorted(self.depths)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.testsize, self.testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.outputIASSF, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.ToTensor()
#         # self.gt_transform = transforms.Compose([
#         #     transforms.Resize((self.trainsize, self.trainsize)),
#         #     transforms.ToTensor()])
#         self.depths_transform = transforms.Compose(
#             [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
#         self.size = len(self.images)
#         self.index = 0
#
#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         image = self.transform(image).unsqueeze(0)
#         gt = self.binary_loader(self.gts[self.index])
#         depth = self.binary_loader(self.depths[self.index])
#         depth = self.depths_transform(depth).unsqueeze(0)
#
#         d = depth[0].numpy()
#         regions = np.digitize(d[0], bins=threshold_multiotsu(d[0]))
#         h, w = d[0].shape
#         bin = np.zeros((3, h, w))
#         for i in range(3):
#             bin[i] = regions == i
#
#         name = self.images[self.index].split('/')[-1]
#         image_for_post = self.rgb_loader(self.images[self.index])
#         image_for_post = image_for_post.resize(gt.size)
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
#         self.index += 1
#         self.index = self.index % self.size
#
#         return image, gt, depth, name, np.array(image_for_post), torch.from_numpy(bin).float()
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def __len__(self):
#         return self.size
#

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

# Reference from BBSNet, Thanks!!!

# several data augumentation strategies
def cv_random_flip(img, label, t):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        t = t.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     t = t.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, t


def randomCrop(image, label, t):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), t.crop(random_region)


def randomRotation(image, label, t):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        t = t.rotate(random_angle, mode)
    return image, label, t


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized t maps for training and test. If you use the normalized t maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, t_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.ts = [t_root + f for f in os.listdir(t_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.ts = sorted(self.ts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.551, 0.619, 0.532], [0.341,  0.360, 0.753])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.ts_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor(), transforms.Normalize([0.241, 0.236, 0.244], [0.208, 0.269, 0.241])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        t = self.rgb_loader(self.ts[index])  # RGBT

        image, gt, t = cv_random_flip(image, gt, t)
        image, gt, t = randomCrop(image, gt, t)
        image, gt, t = randomRotation(image, gt, t)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        t = self.ts_transform(t)

        return image, gt, t

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        ts = []

        for img_path, gt_path, t_path in zip(self.images, self.gts, self.ts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            t = Image.open(t_path)
            if img.size == gt.size and gt.size == t.size:
                images.append(img_path)
                gts.append(gt_path)
                ts.append(t_path)
        self.images = images
        self.gts = gts
        self.ts = ts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, t):
        assert img.size == gt.size and gt.size == t.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   t.resize((w, h),Image.NEAREST)
        else:
            return img, gt, t

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, t_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, t_root, trainsize)
    # print(image_root)
    # print(gt_root)
    # print(t_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, t_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.ts = [t_root + f for f in os.listdir(t_root) if f.endswith('.jpg') or f.endswith('.bmp')]#rgbt
        self.ts = [t_root + f for f in os.listdir(t_root) if f.endswith('.png') or f.endswith('.bmp')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.ts = sorted(self.ts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.551, 0.619, 0.532], [0.341,  0.360, 0.753])])
        self.gt_transform = transforms.ToTensor()
        self.ts_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor(), transforms.Normalize([0.241, 0.236, 0.244], [0.208, 0.269, 0.241])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        t = self.rgb_loader(self.ts[self.index]) # RGBT
        t = self.transform(t).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, t, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
