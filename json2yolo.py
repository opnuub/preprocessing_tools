import json
import os
import cv2
from glob import glob
import shutil
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()

def len_class(files):
    object_list = []
    region_list = []
    object_name_list = []
    region_name_list = []
    for file in files:
        json_path = os.path.join(ROOT_DIR, f'{file}.json')
        f = open(json_path, 'r', encoding='utf-8')
        json_file = json.load(f)
        shapes = json_file["shapes"]
        for shape in shapes:
            group = shape["shape_type"]
            label = shape['label']
            if group == "rectangle":
                if label not in object_list:
                    object_list.append(label)
                if file not in object_name_list:
                    object_name_list.append(file)
            elif group == "polygon":
                if label not in region_list:
                    region_list.append(label)
                if file not in region_name_list:
                    region_name_list.append(file)
        f.close()
    return object_list, region_list, object_name_list, region_name_list

def split_dataset(files, isTest=False):
    test_files = None
    if isTest:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    else:
        trainval_files = files
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
    return train_files, val_files, test_files, files

def create_folders(group, isTest=False):
    train_image = os.path.join(ROOT_DIR, group, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(ROOT_DIR, group, 'train', 'labels')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    valid_image = os.path.join(ROOT_DIR, group, 'valid', 'images')
    if not os.path.exists(valid_image):
        os.makedirs(valid_image)
    valid_label = os.path.join(ROOT_DIR, group, 'valid', 'labels')
    if not os.path.exists(valid_label):
        os.makedirs(valid_label)
    if isTest:
        test_image = os.path.join(ROOT_DIR, group, 'test', 'images')
        if not os.path.exists(test_image):
            os.makedirs(test_image)
        test_label = os.path.join(ROOT_DIR, group, 'test', 'labels')
        if not os.path.exists(test_label):
            os.makedirs(test_label)
        return train_image, train_label, valid_image, valid_label, test_image, test_label
    else:
        return train_image, train_label, valid_image, valid_label

def convert(points, imgsz):
    string = ''
    for point in points:
        dx = point[0] / imgsz[0]
        dy = point[1] / imgsz[1]
        string += ' {} {}'.format(dx, dy)
    return string

def convertBox(point, shape):
    awidth, aheight = shape
    lefttop = point[0]
    rightbottom = point[1]
    width = (rightbottom[0] - lefttop[0])
    length = (rightbottom[1] - lefttop[1])
    rawx = width/2 + lefttop[0]
    rawy = length/2 + lefttop[1]
    return str(rawx/awidth), str(rawy/aheight), str(width/awidth), str(length/aheight) 

def push(files, images, labels, group, suffix='.jpg', isJson=False):
    groups = group.split('/')
    if isJson:
        json_folder = os.path.join(ROOT_DIR, groups[0], groups[1], 'json')
        os.makedirs(json_folder)
    for file in files:
        image = os.path.join(ROOT_DIR, file + suffix)
        label = os.path.join(ROOT_DIR, file + '.txt')
        if isJson:
            json_path = os.path.join(ROOT_DIR, file + '.json')
            shutil.copy(json_path, json_folder)
        shutil.copy(image, images)
        shutil.move(label, labels)

def create_yaml(classes, group, isTest=False):
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
    yamlpath = os.path.join(ROOT_DIR, group, "data.yaml")

    with open(yamlpath, "w+", encoding="utf-8") as f:
        for key,val in desired_caps.items():
            yaml.dump({key:val}, f, default_flow_style=False)

def json2txt(classes, files, list, datatype, suffix):
    for file in files:
        if file in list:
            json_filename = os.path.join(ROOT_DIR, f'{file}.json')
            with open(json_filename, 'r', encoding='utf-8') as f:
                json_file = json.load(f)
                image = os.path.join(ROOT_DIR, f'{file}{suffix}')
                txt_file = open(f'{ROOT_DIR}/{file}.txt', 'w')
                if os.path.exists(image):
                    height, width, _ = cv2.imread(image).shape
                    string = ''
                    if datatype == 'polygon':
                        for shape in json_file['shapes']:
                            label = shape['label']
                            if shape['shape_type'] == datatype:
                                class_id = classes.index(label)
                                if string == '':
                                    string = str(class_id) + convert(shape['points'], (width, height))
                                else:
                                    string = string + '\n' + str(class_id) + convert(shape['points'], (width, height))
                        txt_file.write(string)
                        txt_file.close()
                    elif datatype == 'rectangle':
                        for shape in json_file['shapes']:
                            label = shape['label']
                            if shape['shape_type'] == datatype:
                                class_id = classes.index(label)
                                if string == '':
                                    x, y, w, h = convertBox(shape['points'], (width, height))
                                    string = str(class_id) + f' {x} {y} {w} {h}'
                                else:
                                    x, y, w, h = convertBox(shape['points'], (width, height))
                                    string = string + '\n' + str(class_id) + f' {x} {y} {w} {h}'
                        txt_file.write(string)
                        txt_file.close()

def json2yolo(suffix='.jpg', isTest=False):
    files = glob(ROOT_DIR + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    objectClasses, regionClasses, objectFiles, regionFiles = len_class(files)

    if regionFiles:
        json2txt(regionClasses, files, regionFiles, 'polygon', suffix)
        if isTest:
            rtrain_image, rtrain_label, rvalid_image, rvalid_label, rtest_image, rtest_label = create_folders('polygon', isTest)
            rtrain, rvalid, rtest, rfiles = split_dataset(regionFiles, isTest)
            push(rtrain, rtrain_image, rtrain_label, 'polygon/train', suffix, True)
            push(rvalid, rvalid_image, rvalid_label, 'polygon/valid', suffix, True)
            push(rtest, rtest_image, rtest_label, 'polygon/test', suffix, True)
        else:
            rtrain_image, rtrain_label, rvalid_image, rvalid_label = create_folders('polygon', isTest)
            rtrain, rvalid, _, rfiles = split_dataset(regionFiles, isTest)
            push(rtrain, rtrain_image, rtrain_label, 'polygon/train', suffix, True)
            push(rvalid, rvalid_image, rvalid_label, 'polygon/valid', suffix, True)
        create_yaml(regionClasses, 'polygon', isTest)

    if objectFiles:
        json2txt(objectClasses, files, objectFiles, 'rectangle', suffix)
        if isTest:
            train_image, train_label, valid_image, valid_label, test_image, test_label = create_folders('rectangle', isTest)
            train, valid, test, files = split_dataset(objectFiles, isTest)
            push(train, train_image, train_label, 'rectangle/train', suffix, True)
            push(valid, valid_image, valid_label, 'rectangle/valid', suffix, True)
            push(test, test_image, test_label, 'rectangle/test', suffix, True)
        else:
            train_image, train_label, valid_image, valid_label = create_folders('rectangle', isTest)
            train, valid, _, files = split_dataset(objectFiles, isTest)
            push(train, train_image, train_label, 'rectangle/train', suffix, True)
            push(valid, valid_image, valid_label, 'rectangle/valid', suffix, True)
        create_yaml(objectClasses, 'rectangle', isTest)

if __name__ == '__main__':
    json2yolo('.jpg', True)