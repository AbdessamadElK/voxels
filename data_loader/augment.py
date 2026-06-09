import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Augmentor:
    def __init__(self, crop_size, spatial_aug = True, min_scale=-0.2, max_scale=0.4, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug = spatial_aug
        self.spatial_aug_prob = 0.8
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
            
    
    def spatial_transform(self, voxel1, voxel2, img):
        ht, wd = voxel2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        
        if self.spatial_aug and (np.random.rand() < self.spatial_aug_prob):
            # rescale the images
            voxel1 = cv2.resize(voxel1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            voxel2 = cv2.resize(voxel2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # print('Resized:', voxel1.shape, voxel2.shape, flow.shape, valid.shape)

            img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        
        if self.crop_size is not None:
            margin_y = int(round(35 * scale_y))#downside
            margin_x = int(round(0 * scale_x))#leftside

            y0 = np.random.randint(0, voxel2.shape[0] - self.crop_size[0] - margin_y)
            x0 = np.random.randint(margin_x, voxel2.shape[1] - self.crop_size[1])

            y0 = np.clip(y0, 0, voxel2.shape[0] - self.crop_size[0])
            x0 = np.clip(x0, 0, voxel2.shape[1] - self.crop_size[1])
            
            voxel1 = voxel1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            voxel2 = voxel2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            img = img[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                voxel1 = voxel1[:, ::-1]
                voxel2 = voxel2[:, ::-1]
                img = img[:, ::-1]


            if np.random.rand() < self.v_flip_prob: # v-flip
                voxel1 = voxel1[::-1, :]
                voxel2 = voxel2[::-1, :]
                img = img[::-1, :]

        return voxel1, voxel2, img
    
    def __call__(self, voxel1, voxel2, img):
        voxel1, voxel2, img = self.spatial_transform(voxel1, voxel2, img)
        voxel1 = np.ascontiguousarray(voxel1)
        voxel2 = np.ascontiguousarray(voxel2)
        img = np.ascontiguousarray(img)
        return voxel1, voxel2, img   
                
                        