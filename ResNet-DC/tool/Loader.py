import os
import torch
import cv2
from skimage import io, color
import numpy as np
from torch.utils.data import Dataset
import random


def HorizontalFlip(image):
    image_flip = cv2.flip(image, 1)
    return image_flip


class RandomHorizontallyFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = HorizontalFlip(sample['image'])
            dense = HorizontalFlip(sample['dense'][0])
            dense = dense[np.newaxis, :, :]
            sample['dense'] = dense
        return sample


class Normalize(object):
    def __call__(self, sample):
        image = sample['image']
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.
        image = (image - mean) / std
        sample['image'] = image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, dense = sample['image'], sample['dense']
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = torch.from_numpy(mask).type(torch.LongTensor)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample['image'] = torch.tensor(image).type(torch.FloatTensor)
        sample['dense'] = torch.from_numpy(dense).type(torch.FloatTensor)
        return sample


class Loader(Dataset):
    def __init__(self, img_dir, dense_dir, mask_dir=None, transform=None):
        self.names = []
        for name in os.listdir(img_dir):
            self.names.append(name[:-3])
        self.img_dir = img_dir
        self.dense_dir = dense_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.len = len(self.names)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.names[idx]
        image = io.imread(self.img_dir + name + 'jpg').astype(np.float32)
        if image.ndim == 2:
            image = color.gray2rgb(image)
        dense = np.loadtxt(self.dense_dir + name + 'csv', delimiter=',').astype(np.float32)
        dense = dense[np.newaxis, :, :]
        if self.mask_dir is None:
            sample = {'image': image, 'dense': dense}
        else:
            mask = np.loadtxt(self.mask_dir + name + 'csv', delimiter=',').astype(np.int)
            sample = {'image': image, 'dense': dense, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_min_size(batch):
    min_ht = 576
    min_wd = 768

    for i_sample in batch:

        _, ht, wd = i_sample.shape
        if ht < min_ht:
            min_ht = ht
        if wd < min_wd:
            min_wd = wd
    return min_ht, min_wd


def deNormalize(image):
    image = image.transpose(0, 2, 3, 1)
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
    image = (image * std + mean) * 255.
    image = image.astype(int)
    return image


def random_crop(img, den, mask, dst_size):
    # dst_size: ht, wd
    _, ts_hd, ts_wd = img.shape

    x1 = random.randint(0, ts_wd - dst_size[1])
    y1 = random.randint(0, ts_hd - dst_size[0])
    x2 = x1 + dst_size[1]
    y2 = y1 + dst_size[0]

    if mask is None:
        return img[:, y1:y2, x1:x2], den[:, y1:y2, x1:x2]
    return img[:, y1:y2, x1:x2], den[:, y1:y2, x1:x2], mask[y1:y2, x1:x2]


def my_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # print(batch)
    imgs = [item['image'] for item in batch]
    dens = [item['dense'] for item in batch]
    masks = None
    if 'mask' in batch[0]:
        masks = [item['mask'] for item in batch]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        min_ht, min_wd = get_min_size(imgs)
        cropped_imgs = []
        cropped_dens = []
        cropped_masks = []
        for i_sample in range(len(batch)):
            if masks is None:
                _img, _den = random_crop(imgs[i_sample], dens[i_sample], None, [min_ht, min_wd])

            else:
                _img, _den, _mask = random_crop(imgs[i_sample], dens[i_sample], masks[i_sample], [min_ht, min_wd])
                cropped_masks.append(_mask)
            cropped_imgs.append(_img)
            cropped_dens.append(_den)
        cropped_imgs = torch.stack(cropped_imgs, 0)
        cropped_dens = torch.stack(cropped_dens, 0)
        if masks is None:
            batch = {'image': cropped_imgs, 'dense': cropped_dens}
        else:
            cropped_masks = torch.stack(cropped_masks, 0)
            batch = {'image': cropped_imgs, 'dense': cropped_dens, 'mask': cropped_masks}
        return batch
    raise TypeError((error_msg.format(type(batch[0]))))


if __name__ == '__main__':
    from tool.config import opt
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from tool import vis

    train_dataset = Loader(opt.train_iFname, opt.train_pFname, opt.train_mFname,
                           transform=transforms.Compose([
                               RandomHorizontallyFlip(),
                               ToTensor()])
                           )
    trainLoader = DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=my_collate, shuffle=True)

    # imgs = train_dataset[0]['image']
    # vis.vis_img(imgs.numpy().transpose(1, 2, 0).astype(np.uint), save=False)
    for sample in trainLoader:
        imgs = sample['image'][0].numpy().transpose(1, 2, 0)
        denses = sample['dense'][0].numpy()
        masks = sample['mask'][0].numpy()
        vis.vis_img(imgs.astype(np.int), save=False)
        vis.vis_heat(denses[0], save=False)
        vis.vis_mask(masks, save=False)
        break
