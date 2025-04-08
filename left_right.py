import os
import cv2
from glob import glob

ROOT_DIR = os.getcwd()

left = os.path.join(ROOT_DIR, 'left')
right = os.path.join(ROOT_DIR, 'right')

if not os.path.exists(left):
    os.makedirs(left)
if not os.path.exists(right):
    os.makedirs(right)

files = glob(ROOT_DIR + "\\*.png")
files = [i.replace("\\", "/").split("/")[-1].split(".png")[0] for i in files]

w = 1792

for file in files:
    print(file)
    img = cv2.imread(f'{file}.png')
    left_img = img[:, :w, :]
    right_img = img[:, w:, :]
    cv2.imwrite(f'./left/{file}.png', left_img)
    cv2.imwrite(f'./right/{file}.png', right_img)