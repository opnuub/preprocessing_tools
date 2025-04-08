import json
import os
from glob import glob
import shutil
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()

def split_dataset(files, isTest=True):
    test_files = None
    if isTest:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    else:
        trainval_files = files
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
    return train_files, val_files, test_files

def create_folders(isTest=True):
    train_image = os.path.join(ROOT_DIR, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(ROOT_DIR, 'train', 'labelTxt')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    valid_image = os.path.join(ROOT_DIR, 'valid', 'images')
    if not os.path.exists(valid_image):
        os.makedirs(valid_image)
    valid_label = os.path.join(ROOT_DIR, 'valid', 'labelTxt')
    if not os.path.exists(valid_label):
        os.makedirs(valid_label)
    if isTest:
        test_image = os.path.join(ROOT_DIR, 'test', 'images')
        if not os.path.exists(test_image):
            os.makedirs(test_image)
        test_label = os.path.join(ROOT_DIR, 'test', 'labelTxt')
        if not os.path.exists(test_label):
            os.makedirs(test_label)
        return train_image, train_label, valid_image, valid_label, test_image, test_label
    else:
        return train_image, train_label, valid_image, valid_label

def push(files, images, labels, suffix='.jpg', isJson=True):
    if isJson:
        json_folder = os.path.join(ROOT_DIR, 'json')
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
    for file in files:
        image = os.path.join(ROOT_DIR, file + suffix)
        label = os.path.join(ROOT_DIR, file + '.txt')
        if isJson:
            json_path = os.path.join(ROOT_DIR, file + '.json')
            shutil.move(json_path, json_folder)
        shutil.move(image, images)
        shutil.move(label, labels)

def create_yaml(classes, isTest=True):
    nc = len(classes)
    if not isTest:
        desired_caps = {
            'train': 'train/images',
            'val': 'valid/images',
            'nc': nc,
            'names': classes
        }
    else:
        desired_caps = {
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': nc,
            'names': classes
        }
    yamlpath = os.path.join(ROOT_DIR, "data.yaml")

    with open(yamlpath, "w+", encoding="utf-8") as f:
        for key,val in desired_caps.items():
            yaml.dump({key:val}, f, default_flow_style=False)

def process(files):
    classes = []
    for file in files:
        with open(f'{file}.json', 'r') as f:
            string = ''
            json_file = json.load(f)
            txt_file = open(f'{file}.txt', 'w')
            for shape in json_file['shapes']:
                label = shape['label']
                if label not in classes:
                    classes.append(label)
                points = shape['points']
                print(points)
                for point in points:
                    print(point)
                    string = string + f'{point[0]:.5f} {point[1]:.5f} '
                string = string + f'{label} 0\n'
            txt_file.write(string)
            txt_file.close()
    return classes

if __name__ == '__main__':

    isTest, isJson, suffix = True, True, '.jpg'

    files = glob(ROOT_DIR + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    
    classes = process(files)
    create_yaml(classes, isTest)
    
    if isTest:
        train_image, train_label, val_image, val_label, test_image, test_label = create_folders(isTest)
        train_files, val_files, test_files = split_dataset(files, isTest)
        push(train_files, train_image, train_label)
        push(test_files, test_image, test_label)
        push(val_files, val_image, val_label)
    else:
        train_image, train_label, val_image, val_label = create_folders(isTest)
        train_files, val_files, _ = split_dataset(files, isTest)
        push(train_files, train_image, train_label)
        push(val_files, val_image, val_label)