import numpy as np
import torch
import torch.utils.data as data

import random
import os
from pathlib import Path
import glob

import imageio.v2 as imageio
import matplotlib

from .augment import Augmentor

DATA_SPLIT = {
    'train' : ['zurich_city_00_a',
               'zurich_city_00_b',
               'zurich_city_01_a',
               'zurich_city_01_b',
               'zurich_city_01_c',
               'zurich_city_01_d',
               'zurich_city_01_e',
               'zurich_city_01_f',
               'zurich_city_02_a',
               'zurich_city_02_b',
               'zurich_city_02_c'],
    
    'val'   : ['interlaken_00_d',
               'thun_00_a'],

    # 'val'   : ['interlaken_00_c',
    #            'interlaken_00_d',
    #            'interlaken_00_e',
    #            'interlaken_00_f',
    #            'interlaken_00_g',
    #            'thun_00_a']
}

DATA_SAMPLES = [('zurich_city_01_a', 2),
                ('interlaken_00_d', 0),
                ('thun_00_a', 24)]

def get_seq_idx_from_path(path: Path) -> str | None:
    keywords = {"zurich", "thun", "interlaken"}

    seq = None
    try:
        idx = int(path.stem)
    except ValueError:
        print('Failed extracting sample index from file name : {}'.format(path.stem))
        idx = None

    for parent in path.parents:
        name_lower = parent.name.lower()
        if any(keyword in name_lower for keyword in keywords):
            seq = parent.name
            break
        
    if seq is None:
        print('Failed extracting sequence name from file path : {}'.format(path))


    return (seq, idx)

def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    #img = img / (img.max() + 1e-8)

    img_ = img.flatten()
    max_value = np.percentile(img_, q=98)
    img = img / (max_value + 1e-8)
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

class DSECfull(data.Dataset):
    def __init__(self, phase, crop:bool = True, flip:bool = True, spatial_aug:bool = True):
        assert phase in ["train", "val", "trainval", "test", "sample"]

        self.init_seed = False
        self.phase = phase
        self.voxels = []
        self.flows = []

        ### Please change the root to satisfy your data saving setting.
        root = './data/dsec_v2e_events'
        if phase == 'train' or phase == 'trainval':
            self.root = os.path.join(root, 'trainval')

            self.augment = crop or flip or spatial_aug
            crop_size = [288, 384] if crop else None
            self.augmentor = Augmentor(crop_size, do_flip=flip, spatial_aug = spatial_aug)

        elif phase == 'val' or phase == 'sample':
            self.root = os.path.join(root, 'trainval')
            self.augment = False

        elif phase == 'test':
            self.root = os.path.join(root, 'test')
            self.augment = False
        

        # GT Event files
        self.voxels_gt = glob.glob(os.path.join(self.root, '*', 'real_voxel_grids', '*.npz'))

        # Input event files
        self.voxels = glob.glob(os.path.join(self.root, '*', 'v2e_voxel_grids', '*.npz'))

        # Image files
        self.images = glob.glob(os.path.join(self.root, '*', 'images', '*.png'))

        # Event Images
        # self.event_images = glob.glob(os.path.join(self.root, '*', 'event_images', '*.png'))

        self.voxels.sort()
        self.voxels_gt.sort()
        self.images.sort()

        if self.phase == 'train' or self.phase == 'val':
            self.filter_files_by_phase()

        if self.phase == 'sample':
            self.filter_sample_files()


    def filter_files_by_phase(self):
        self.voxels = [f for f in self.voxels if get_seq_idx_from_path(Path(f))[0] in DATA_SPLIT[self.phase]]
        self.voxels_gt = [f for f in self.voxels_gt if get_seq_idx_from_path(Path(f))[0] in DATA_SPLIT[self.phase]]
        self.images = [f for f in self.images if get_seq_idx_from_path(Path(f))[0] in DATA_SPLIT[self.phase]]


    def filter_sample_files(self):
        self.voxels = [f for f in self.voxels if get_seq_idx_from_path(Path(f)) in DATA_SAMPLES]
        self.voxels_gt = [f for f in self.voxels_gt if get_seq_idx_from_path(Path(f)) in DATA_SAMPLES]
        self.images = [f for f in self.images if get_seq_idx_from_path(Path(f)) in DATA_SAMPLES]

    
    def get_data_sample(self, index):
        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True
        
        #events
        voxel = np.load(self.voxels[index])['voxel']
        voxel = voxel.transpose(1, 2, 0)

        voxel_gt = np.load(self.voxels_gt[index])['voxel']
        voxel_gt = voxel_gt.transpose(1, 2, 0)

        #image
        img1 = imageio.imread(self.images[index])
        
        #data augmentation
        if self.augment:
            voxel, voxel_gt, img1 = self.augmentor(voxel, voxel_gt, img1)

        #to Tensor
        voxel = torch.from_numpy(voxel).permute(2, 0, 1).float()
        voxel_gt = torch.from_numpy(voxel_gt).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        
        return voxel, voxel_gt, img1


    def __getitem__(self, index):
        # if self.phase != 'test':
        #     img_file = self.images[index]
        #     img_idx = int(os.path.basename(img_file).split('.')[0])
        #     next_img_file = os.path.join(os.path.dirname(img_file), "{:06d}.png".format(img_idx + 2))
        #     if not os.path.exists(next_img_file):
        #         try:
        #             return self.get_data_sample(index + 1)
        #         except IndexError:
        #             return self.__getitem__(random.randint(0, len(self)-1))
            
        return self.get_data_sample(index)
        

    def __len__(self):
        return len(self.voxels)
    
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_data_loader(phase, batch_size, num_workers, crop = True, flip = True, spatial_aug = True, init_seed = 1):
    g = torch.Generator()
    g.manual_seed(init_seed)

    dset = DSECfull(phase, crop, flip, spatial_aug)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        drop_last=True)
    
    return loader


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    dset = DSECfull('test', crop=False, flip=False, spatial_aug=False)

    items = dset.get_data_sample(415)

    print(len(items))
    quit()

    for item in dset:
        v1, v2, img1, img2, *_, coords = item
        print(coords, v1.shape, img1.shape, sep = "\t")


    v1, v2, flow, valid, img1, img2, depth = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape, img1.shape, img2.shape, depth.shape)


    depth = depth[0].numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(depth)
    
    plt.subplot(1, 2, 2)
    plt.imshow(img1.permute(1, 2, 0).numpy().astype('uint8'))
    
    plt.show()

