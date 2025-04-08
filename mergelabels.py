import json
import os
import cv2
import shutil
import yaml
from glob import glob
from sklearn.model_selection import train_test_split

# This program only works on json files
# Splits dataset into bounding box and polygons first

def split_classes(files):
    bbox_classes = []
    bbox_files = []
    polygon_classes = []
    polygon_files= []
    for file in files:
        json_path = os.path.join(ROOT_DIR, f'{file}.json')
        with open(json_path, 'r') as f:
            json_file = json.load(f)
            for item in json_file['shapes']:
                if item['shape_type'] == 'rectangle':
                    if file not in bbox_files:
                        bbox_files.append(file)
                    if item['label'] not in bbox_classes:
                        bbox_classes.append(item['label'])
                else:
                    if file not in polygon_files:
                        polygon_files.append(file)
                    if item['label'] not in polygon_classes:
                        polygon_classes.append(item['label'])
    return bbox_classes, bbox_files, polygon_classes, polygon_files

# classes can be bbox_classes or polygon_classes
def get_replacement(classes, label):
    print(classes)
    dictionary = {}
    replacements = []
    len_classes = int(input(f'Number of unique classes for {label}: '))
    for _ in range(len_classes):
        replacement = input('\nReplacement: ')
        replacements.append(replacement)
        a = ''
        print('Input * to stop')
        while a != '*':
            a = input('Substitute: ')
            dictionary[a] = replacement
    return dictionary, replacements

def create_yaml(classes, label, isTest=True):
    nc = len(classes) # using replacements
    if not isTest:
        desired_caps = {
            'train': 'train',
            'val': 'valid',
            'nc': nc,
            'names': classes
        }
    else:
        desired_caps = {
            'train': 'train',
            'val': 'valid',
            'test': 'test',
            'nc': nc,
            'names': classes
        }
    yamlpath = os.path.join(ROOT_DIR, label, "data.yaml")
    with open(yamlpath, "w+", encoding="utf-8") as f:
        for key,val in desired_caps.items():
            yaml.dump({key:val}, f, default_flow_style=False)


def split_dataset(files, isTest=True):
    test_files = None
    if isTest:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=42)
    else:
        trainval_files = files
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=42)
    return train_files, val_files, test_files

def create_folders(label):
    train_image = os.path.join(ROOT_DIR, label, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(ROOT_DIR, label, 'train', 'labels')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    valid_image = os.path.join(ROOT_DIR, label, 'valid', 'images')
    if not os.path.exists(valid_image):
        os.makedirs(valid_image)
    valid_label = os.path.join(ROOT_DIR, label, 'valid', 'labels')
    if not os.path.exists(valid_label):
        os.makedirs(valid_label)
    test_image = os.path.join(ROOT_DIR, label, 'test', 'images')
    if not os.path.exists(test_image):
        os.makedirs(test_image)
    test_label = os.path.join(ROOT_DIR, label, 'test', 'labels')
    if not os.path.exists(test_label):
        os.makedirs(test_label)
    return train_image, train_label, valid_image, valid_label, test_image, test_label

def push(files, datatype, images_folder, labels_folder, suffix='jpg'):
    """
    Copies the image and move its annotation to the target folders
    :param files: List of file paths of corresponding datatype
    :param datatype: polygon/ bbox
    :return: None
    """
    for file in files:
        image = os.path.join(ROOT_DIR, f'{file}.{suffix}')
        label = os.path.join(ROOT_DIR, f'{datatype}_{file}.txt')
        if os.path.exists(image) and os.path.exists(label):
            shutil.copy(image, images_folder)
            old_image_path = os.path.join(images_folder, f'{file}.{suffix}')
            new_image_path = os.path.join(images_folder, f'{datatype}_{file}.{suffix}')
            os.rename(old_image_path, new_image_path)
            shutil.move(label, labels_folder)

def convert(points, imgSize):
    """
    Normalise list of raw points and returns string of all coordinates
    :param points: List of points
    :param imgSize: Tuple of image shape
    :return:  
    """
    string = ''
    for point in points:
        dx = point[0] / imgSize[0]
        dy = point[1] / imgSize[1]
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

def json2txt(classes, files, group, dictionary, suffix):
    lst = [0] * len(classes)
    for file in files:
        json_path = os.path.join(ROOT_DIR, f'{file}.json')
        with open(json_path, 'r') as f:
            json_file = json.load(f)
            image_path = os.path.join(ROOT_DIR, f'{file}.{suffix}')
            txt_file = open(f'{ROOT_DIR}/{group}_{file}.txt', 'w')
            if os.path.exists(image_path):
                height, width, _ = cv2.imread(image_path).shape
                string = ''
                if group == 'polygon':
                    for shape in json_file['shapes']:
                        label = shape['label']
                        if shape['shape_type'] == group:
                            replacement = dictionary[label]
                            class_id = classes.index(replacement)
                            lst[class_id] = lst[class_id] + 1
                            if string == '':
                                string = str(class_id) + convert(shape['points'], (width, height))
                            else:
                                string = string + '\n' + str(class_id) + convert(shape['points'], (width, height))
                    txt_file.write(string)
                    txt_file.close()
                else:
                    for shape in json_file['shapes']:
                        label = shape['label']
                        if shape['shape_type'] == 'rectangle':
                            replacement = dictionary[label]
                            class_id = classes.index(replacement)
                            lst[class_id] = lst[class_id] + 1
                            if string == '':
                                x, y, w, h = convertBox(shape['points'], (width, height))
                                string = str(class_id) + f' {x} {y} {w} {h}'
                            else:
                                x, y, w, h = convertBox(shape['points'], (width, height))
                                string = string + '\n' + str(class_id) + f' {x} {y} {w} {h}'
                    txt_file.write(string)
                    txt_file.close()
    print(group, lst)

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    all_files = glob(ROOT_DIR + "\\*.json")
    all_files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in all_files]
    suffix = input('jpg/png/jpeg: ')
    
    bbox_classes, bbox_files, polygon_classes, polygon_files = split_classes(all_files)
    bbox_dict, bbox_labels = get_replacement(bbox_classes, 'bbox') # bbox_labels are in .yaml file
    polygon_dict, polygon_labels = get_replacement(polygon_classes, 'polygon')

    bbox_train_image, bbox_train_label, bbox_valid_image, bbox_valid_label, bbox_test_image, bbox_test_label = create_folders('bbox')
    polygon_train_image, polygon_train_label, polygon_valid_image, polygon_valid_label, polygon_test_image, polygon_test_label = create_folders('polygon')
    create_yaml(bbox_labels, 'bbox')
    create_yaml(polygon_labels, 'polygon')
    json2txt(bbox_labels, bbox_files, 'bbox', bbox_dict, suffix)
    json2txt(polygon_labels, polygon_files, 'polygon', polygon_dict, suffix)

    bbox_train, bbox_valid, bbox_test = split_dataset(bbox_files)
    polygon_train, polygon_valid, polygon_test = split_dataset(polygon_files)
    push(bbox_train, 'bbox', bbox_train_image, bbox_train_label, suffix)
    push(bbox_valid, 'bbox', bbox_valid_image, bbox_valid_label, suffix)
    push(bbox_test, 'bbox', bbox_test_image, bbox_test_label, suffix)
    push(polygon_train, 'polygon', polygon_train_image, polygon_train_label, suffix)
    push(polygon_valid, 'polygon', polygon_valid_image, polygon_valid_label, suffix)
    push(polygon_test, 'polygon', polygon_test_image, polygon_test_label, suffix)