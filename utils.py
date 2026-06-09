import logging
import coloredlogs
import os
import numpy
import cv2

def get_logger(log_path):
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path)
    # log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger


def writer_add_features(tensor_feat):
    feat_img = tensor_feat.detach().cpu().numpy()
    # img_grid = self.make_grid(feat_img)
    feat_img = numpy.sum(feat_img,axis=0)
    feat_img = feat_img -numpy.min(feat_img)
    img_grid = 255*feat_img/numpy.max(feat_img)
    img_grid = numpy.array(img_grid, dtype=numpy.uint8)
    img_grid = cv2.applyColorMap(img_grid, cv2.COLORMAP_JET)
    return img_grid


def writer_add_features_normalized(tensor_feat, vmin=None, vmax=None):
    feat_img = tensor_feat.detach().cpu().numpy()

    # Sum channels into one 2D map
    feat_img = numpy.sum(feat_img, axis=0)

    # Use fixed range if provided
    if vmin is None:
        vmin = numpy.min(feat_img)
    if vmax is None:
        vmax = numpy.max(feat_img)

    # Clip values outside the fixed colorbar range
    feat_img = numpy.clip(feat_img, vmin, vmax)

    # Normalize using fixed range
    denom = vmax - vmin
    if denom == 0:
        img_grid = numpy.zeros_like(feat_img, dtype=numpy.uint8)
    else:
        img_grid = 255 * (feat_img - vmin) / denom
        img_grid = img_grid.astype(numpy.uint8)

    # Apply colormap
    # img_grid = cv2.applyColorMap(img_grid, cv2.COLORMAP_JET)

    # Or keep it gray
    img_grid = numpy.dstack(3*[img_grid[...,None]])

    return img_grid