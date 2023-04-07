from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, read_mask, scale=1, mask_suffix=''):

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.read_mask = read_mask
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def do_transpose(cls, img_nd):
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans


    def __getitem__(self, i):
        name = self.ids[i]  #idx = self.ids[i]
        if self.read_mask:
           mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))   #mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = list(self.imgs_dir.glob(name + '.*'))    #img_file = glob(self.imgs_dir + idx + '.*')

        #print(mask_file[0])
        #print(img_file[0])

        if self.read_mask != None:
          assert len(mask_file) == 1, \
              f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
          mask = Image.open(mask_file[0])

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
 
        img = Image.open(img_file[0])

        if self.read_mask != None:
          assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.read_mask == 'rgb':
          #print('Calling inverse map rgb...')
          mask = kitti_inverse_map(np.array(mask, dtype=np.int32))
          #print('Ok inverse map rgb...')
        if self.read_mask == 'gray':
          #print('Calling inverse map gray...')
          mask = kitti_inverse_map_1channel(np.array(mask, dtype=np.int32))
          #print('Ok inverse map gray...')

        #print(np.array(mask))

        img = self.preprocess(img, self.scale)

        if self.read_mask != None:
          #mask = mask[0:370, 0:1224]   #(370, 1224, 3)
          mask_torch = torch.from_numpy(mask).type(torch.IntTensor)
        else:
          mask_torch = None

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': mask_torch
        }
