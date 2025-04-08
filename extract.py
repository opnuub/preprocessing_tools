import json
import os
from glob import glob
import shutil

ROOT_DIR = os.getcwd()
files = glob(ROOT_DIR + "\\*.json")
files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

destination = 'C:/Users/colel/Desktop/Mid_Feb_Batch'
for file in files:
    image_path = os.path.join(ROOT_DIR, f'{file}.jpg')
    json_path = os.path.join(ROOT_DIR, f'{file}.json')
    shutil.move(image_path, destination)
    shutil.move(json_path, destination)