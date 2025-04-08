import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
from glob import glob
import os
import sys
import tifffile

def find_occ_mask(disp_left, disp_right):
    """
    find occlusion map
    1 indicates occlusion
    disp range [0,w]
    """
    w = 1280

    # # left occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disp_left

    # 1. negative locations will be occlusion
    occ_mask_l = right_shifted <= 0

    # 2. wrong matches will be occlusion
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype(np.int)
    disp_right_selected = np.take_along_axis(disp_right, right_shifted,
                                             axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_right_selected - disp_left) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_right_selected <= 0.0] = False
    wrong_matches[disp_left <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_l] = True  # apply case 1 occlusion to case 2
    occ_mask_l = wrong_matches

    # # right occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    left_shifted = coord + disp_right

    # 1. negative locations will be occlusion
    occ_mask_r = left_shifted >= w

    # 2. wrong matches will be occlusion
    left_shifted[occ_mask_r] = 0  # set negative locations to 0
    left_shifted = left_shifted.astype(np.int)
    disp_left_selected = np.take_along_axis(disp_left, left_shifted,
                                            axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_left_selected - disp_right) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_left_selected <= 0.0] = False
    wrong_matches[disp_right <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_r] = True  # apply case 1 occlusion to case 2
    occ_mask_r = wrong_matches

    return occ_mask_l, occ_mask_r

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)
    image.tofile(file)

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    left_dir = os.path.join(ROOT_DIR, 'disparity_left')
    right_dir = os.path.join(ROOT_DIR, 'disparity_right')
    left_files = glob(left_dir + "\\*.tiff")
    right_files = glob(right_dir + "\\*.tiff")
    left_files = [i.replace("\\", "/").split("/")[-1].split(".tiff")[0] for i in left_files]
    right_files = [i.replace("\\", "/").split("/")[-1].split(".tiff")[0] for i in right_files]

    disp_left = os.path.join(ROOT_DIR, 'disp_left')
    disp_right = os.path.join(ROOT_DIR, 'disp_right')
    occ_folder = os.path.join(ROOT_DIR, 'occlusion')
    if not os.path.exists(disp_left):
        os.makedirs(disp_left)
    if not os.path.exists(disp_right):
        os.makedirs(disp_right)
    if not os.path.exists(occ_folder):
        os.makedirs(occ_folder)

    for left_file in left_files:
        im = tifffile.imread(f'disparity_left/{left_file}.tiff')
        im = np.array(im, dtype=np.float32)
        #writePFM(f'disp_left/{left_file}.pfm', im)
        plt.imsave(f'disp_left/{left_file}.png', im)
        img = tifffile.imread(f'disparity_right/{left_file}.tiff', dtype=np.float32)
        img = np.array(img, dtype=np.float32)
        #writePFM(f'disp_right/{left_file}.pfm', img)
        plt.imsave(f'disp_right/{left_file}.png', img)
        
        occ_mask_l, occ_mask_r = find_occ_mask(im, img)
        plt.imsave(f'occlusion/{left_file}.png', occ_mask_l)