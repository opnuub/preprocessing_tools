import xml.etree.ElementTree as ET
import os
import cv2
from glob import glob
import shutil
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()

def convert(points):
    string = ''
    for point in points:
        dx = point[0]
        dy = point[1]
        string += f' {dx} {dy}'
    return string

def push(files, images, labels):
    for file in files:
        image = os.path.join(ROOT_DIR, file + '.jpg')
        label = os.path.join(ROOT_DIR, file + '.txt')
        shutil.move(image, images)
        shutil.move(label, labels)

def split_dataset(files, isTest=False):
    test_files = None
    if isTest:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    else:
        trainval_files = files
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
    return train_files, val_files, test_files

def create_folders(isTest=False):
    train_image = os.path.join(ROOT_DIR, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(ROOT_DIR, 'train', 'labels')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    valid_image = os.path.join(ROOT_DIR, 'valid', 'images')
    if not os.path.exists(valid_image):
        os.makedirs(valid_image)
    valid_label = os.path.join(ROOT_DIR, 'valid', 'labels')
    if not os.path.exists(valid_label):
        os.makedirs(valid_label)
    if isTest:
        test_image = os.path.join(ROOT_DIR, 'test', 'images')
        if not os.path.exists(test_image):
            os.makedirs(test_image)
        test_label = os.path.join(ROOT_DIR, 'test', 'labels')
        if not os.path.exists(test_label):
            os.makedirs(test_label)
        return train_image, train_label, valid_image, valid_label, test_image, test_label
    else:
        return train_image, train_label, valid_image, valid_label

def xml2txt(files):
    ids=[]

    for file in files:
        dictionary = {}
        count = 0 # Abnormaly detection
        
        tree = ET.parse(f'{file}.xml')
        root = tree.getroot()
        imgsz = root.find('imagesize')
        xx, yy = float(imgsz.find('ncols').text), float(imgsz.find('nrows').text)

        for object in root.findall('object'):
            lst = []
            idx = object.find('id').text
            if idx not in ids:
                ids.append(idx)    
            polygon = object.find('polygon')
            for pts in polygon.findall('pt'):
                x, y = pts.find('x').text, pts.find('y').text
                try:
                    x, y = float(x), float(y)
                    x, y = x/xx, y/yy
                except Exception:
                    print(x, y)
                lst.append((x, y))
            if idx in dictionary:
                count += 1
                idx = idx + f'_{count}'
            dictionary[idx] = lst
        
        with open(f'{ROOT_DIR}/{file}.txt', 'w') as f:
            if count == 0:
                string = ''
                for idxx in dictionary:
                    if string == '':
                        string = idxx + convert(dictionary[idxx])
                    else:
                        string = string + '\n' + idxx + convert(dictionary[idxx])
            else:
                string = ''
                for idxx in dictionary:
                    if '_' in idxx:
                        id_real = idxx.split('_')[0]
                    if string == '':
                        string = id_real + convert(dictionary[idxx])
                    else:
                        string = string + '\n' + id_real + convert(dictionary[idxx])
            f.write(string)
    return ids

def create_yaml(classes, isTest=False):
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

def extract(videopath):
    vid = cv2.VideoCapture(videopath)
    count = 0
    ret, image = vid.read()
    string = len(str(count))
    while ret:
        cv2.imwrite(f'{ROOT_DIR}/frame_{(6-string)*"0"}{count}.jpg', image)
        count += 1
        string = len(str(count))
        ret, image = vid.read()

def xml2yolo(videopath, isTest=True):
    files = glob(ROOT_DIR + "\\*.xml")
    files = [i.replace("\\", "/").split("/")[-1].split(".xml")[0] for i in files]

    ids = xml2txt(files)
    extract(videopath)

    if isTest:
        train_image, train_label, valid_image, valid_label, test_image, test_label = create_folders(isTest=True)
        train, valid, test = split_dataset(files, isTest=True)
        push(train, train_image, train_label)
        push(valid, valid_image, valid_label)
        push(test, test_image, test_label)
    else:
        train_image, train_label, valid_image, valid_label = create_folders(isTest=False)
        train, valid, test = split_dataset(files, isTest=False)
        push(train, train_image, train_label)
        push(valid, valid_image, valid_label)
    
    create_yaml(ids, isTest)


if __name__ == '__main__':
    xml2yolo('vid.mp4', True)