import base64
from glob import glob
import json
import cv2
import os
import shutil
import yaml
import labelme
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()

def effectiveFramesAndClasses(annotations):
    # Annotations are list of json filename
    allObjectFrames = []
    allRegionFrames = []
    objectClasses = {}
    regionClasses = {}
    for annotation in annotations:
        f = open(annotation, 'r')
        jsonf = json.load(f)
        objectList = jsonf['annotation']['objectAnnotationListMap']
        regionList = jsonf['annotation']['regionAnnotationListMap']
        objectFrames = [int(frame) for frame in objectList.keys()]
        regionFrames = [int(frame) for frame in regionList.keys()]
        allObjectFrames.append(objectFrames)
        allRegionFrames.append(regionFrames)
        for objectFrame in objectFrames:
            for object in objectList[str(objectFrame)]:
                if object['labelId'] not in objectClasses:
                    objectClasses[object['labelId']] = 1
        for regionFrame in regionFrames:
            for region in regionList[str(regionFrame)]:
                if region['labelId'] not in regionClasses:
                    regionClasses[region['labelId']] = 1
        f.close()
    allObjectClasses = [okey for okey in objectClasses.keys()]
    allRegionClasses = [rkey for rkey in regionClasses.keys()]
    return allObjectFrames, allObjectClasses, allRegionFrames, allRegionClasses

def createFolders(datatype):
    """
    Create subfolders for training, validation and testing inside folder of name 'datatype'
    :param datatype: Name of folder
    :return:
    """
    label_path = ROOT_DIR
    train_image = os.path.join(label_path, datatype, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(label_path, datatype, 'train', 'labels')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    val_image = os.path.join(label_path, datatype, 'valid', 'images')
    if not os.path.exists(val_image):
        os.makedirs(val_image)
    val_label = os.path.join(label_path, datatype, 'valid', 'labels')
    if not os.path.exists(val_label):
        os.makedirs(val_label)
    test_image = os.path.join(label_path, datatype, 'test', 'images')
    if not os.path.exists(test_image):
        os.makedirs(test_image)
    test_label = os.path.join(label_path, datatype, 'test', 'labels')
    if not os.path.exists(test_label):
        os.makedirs(test_label)
    return train_image, train_label, val_image, val_label, test_image, test_label

def extractFrames(oFrames, rFrames, videopath, index):
    """
    Extract labelled frames from video
    :param frames: List of frames needed to be extracted
    :param fps: Target FPS in JSON annotation file
    :param vidopath: Path of input video
    :param outputpath: Output folder for extracted frames
    :return: 
    """
    vid = cv2.VideoCapture(videopath)
    objectFrames = []
    regionFrames = []
    for i in oFrames:
        vid.set(cv2.CAP_PROP_POS_MSEC, i*100)
        ret, image = vid.read()
        if ret:
            cv2.imwrite(f'{ROOT_DIR}/object_box/{index}_{i}.jpg', image)
            objectFrames.append(i)
    for j in rFrames:
        vid.set(cv2.CAP_PROP_POS_MSEC, j*100)
        ret, image = vid.read()
        if ret:
            cv2.imwrite(f'{ROOT_DIR}/region_polygon/{index}_{j}.jpg', image)
            regionFrames.append(j)

    return objectFrames, regionFrames

def createBoxTxt(files, effective, raw, shape, ifJson, classes, index):
    for file in files:

        if ifJson:
            path = f"{ROOT_DIR}/object_box/{index}_{file}.jpg"
            data = labelme.LabelFile.load_image_file(path)
            image_data = base64.b64encode(data).decode('utf-8')
            dictionary = {
                "version": "5.0.5",
                "flags": {},
                "shapes": [],
                "imagePath": path,
                "imageData": image_data,
                "imageHeight": shape[1],
                "imageWidth": shape[0]
            }

        image_filename = os.path.join(ROOT_DIR, f'object_box/{index}_{file}.jpg')
        output = open(f'{ROOT_DIR}/object_box/{index}_{file}.txt', 'w')
        if os.path.exists(image_filename):
            string = ''
            val = raw[str(file)]
            for point in val:
                label = effective.index(point['labelId'])
                x = point['x']
                y = point['y']
                w = point['width']
                h = point['height']
                
                if ifJson:  
                    rawPoints = [[x, y], [x+w, y+h]]
                    json_shape = {
                        "label": classes[point['labelId']],
                        "points": rawPoints,
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    a = dictionary['shapes']
                    a.append(json_shape)
                    dictionary['shapes'] = a

                dx = (x+w/2) / shape[0]
                dy = (y+h/2) / shape[1]
                dw = w / shape[0]
                dh = h / shape[1]
                if string == '':
                    string = f'{label} {dx} {dy} {dw} {dh}'
                else:
                    string += f'\n{label} {dx} {dy} {dw} {dh}'
            output.write(string)
            output.close()
            if ifJson:
                with open(f'{ROOT_DIR}/object_box/{index}_{file}.json', 'w') as out:
                    json.dump(dictionary, out)

def convert(points, shape):
    """
    Normalise list of raw points and returns string of all coordinates
    :param points: List of points
    :param imgSize: Tuple of image shape
    :return:  
    """
    string = ''
    for point in points:
        dx = point['x'] / shape[0]
        dy = point['y'] / shape[1]
        string += ' {} {}'.format(dx, dy)
    return string

def createRegionTxt(files, effective, raw, shape, ifJson, classes, index):
    for file in files:
        if ifJson:
            path = f"{ROOT_DIR}/region_polygon/{index}_{file}.jpg"
            data = labelme.LabelFile.load_image_file(path)
            image_data = base64.b64encode(data).decode('utf-8')
            dictionary = {
                "version": "5.0.5",
                "flags": {},
                "shapes": [],
                "imagePath": path,
                "imageData": image_data,
                "imageHeight": shape[1],
                "imageWidth": shape[0]
            }
        image_filename = os.path.join(ROOT_DIR, f'region_polygon/{index}_{file}.jpg')
        output = open(f'{ROOT_DIR}/region_polygon/{index}_{file}.txt', 'w')
        if os.path.exists(image_filename):
            string = ''
            val = raw[str(file)]
            for point in val:
                label = effective.index(point['labelId'])
                points = point['pointList']
                
                if ifJson:
                    rawPoints = []
                    for rawPoint in points:
                        rawPoints.append([rawPoint['x'], rawPoint['y']])
                    json_shape = {
                        "label": classes[point['labelId']],
                        "points": rawPoints,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    a = dictionary['shapes']
                    a.append(json_shape)
                    dictionary['shapes'] = a

                if string == '':
                    string = str(label) + convert(points, shape)
                else:
                    string = string + '\n' + str(label) + convert(points, shape)
            output.write(string)
            output.close()
            with open(f'{ROOT_DIR}/region_polygon/{index}_{file}.json', 'w') as out:
                json.dump(dictionary, out)

def split_dataset(datatype):
    files = glob(f'{ROOT_DIR}\\{datatype}' + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
    return train_files, val_files, test_files

def push(files, images, labels, datatype, ifJson):
    if ifJson:
        jsonfolder1 = images.split('\\')[:-1]
        jsonfolder1.append('json')
        jsonfolder = "\\".join(jsonfolder1)
        os.makedirs(jsonfolder)
    for file in files:
        image = os.path.join(ROOT_DIR, f'{datatype}/{file}.jpg')
        label = os.path.join(ROOT_DIR, f'{datatype}/{file}.txt')
        if ifJson:
            jsonfile = os.path.join(ROOT_DIR, f'{datatype}/{file}.json')
            try:
                shutil.move(jsonfile, jsonfolder)
            except OSError as error:
                print(error)
        if not os.path.exists(os.path.join(images, f'{file}.jpg')):
            try:
                shutil.move(image, images)
                shutil.move(label, labels)
            except OSError as error:
                print(error)

def createYAML(rawClasses, effectiveClasses, folder):
    nc = len(effectiveClasses)
    classes = [rawClasses[i] for i in effectiveClasses]
    desired_caps = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': nc,
        'names': classes
    }
    yamlpath = os.path.join(ROOT_DIR, f'{folder}/data.yaml')
    with open(yamlpath, 'w+', encoding='utf-8') as f:
        for key, val in desired_caps.items():
            yaml.dump({key:val}, f, default_flow_style=False)

def vidsToYolo(videos, annotations):
    # Videos type:list, Annotations type:list
    if len(videos) != len(annotations):
        return None
    objectFrames, objectClasses, regionFrames, regionClasses = effectiveFramesAndClasses(annotations=annotations)
    if objectClasses:
        oTrainImage, oTrainLabel, oValidImage, oValidLabel, oTestImage, oTestLabel = createFolders('object_box')
    if regionClasses:
        rTrainImage, rTrainLabel, rValidImage, rValidLabel, rTestImage, rTestLabel = createFolders('region_polygon')

    ifJson=True #Testing purposes
    for index in range(len(videos)):
        print(f'Extracting images from video {index+1}...')
        video = videos[index]
        annotation = annotations[index]
        f = open(annotation, 'r')   
        json_file = json.load(f)
        rawClasses = [i['name'] for i in json_file['config']['objectLabelData']]
        shape = (json_file['annotation']['video']['width'], json_file['annotation']['video']['height'])
        o, r = extractFrames(objectFrames[index], regionFrames[index], video, index)
        objectFrames[index] = o
        regionFrames[index] = r 
        
        print(f'Extraction complete for video {index+1}, creating text annotations...\n')
        if objectFrames[index]:
            createBoxTxt(objectFrames[index], objectClasses, json_file['annotation']['objectAnnotationListMap'], shape, ifJson, rawClasses, index)
        if regionFrames[index]:
            createRegionTxt(regionFrames[index], regionClasses, json_file['annotation']['regionAnnotationListMap'], shape, ifJson, rawClasses, index)
        print(f'Converstion done for video {index+1}\n')
        f.close()
    
    if objectClasses:
        oTrain, oValid, oTest = split_dataset('object_box')
        createYAML(rawClasses, objectClasses, 'object_box')
        push(oTrain, oTrainImage, oTrainLabel, 'object_box', ifJson)
        push(oTest, oTestImage, oTestLabel, 'object_box', ifJson)
        push(oValid, oValidImage, oValidLabel, 'object_box', ifJson)
        
    if regionClasses:
        rTrain, rValid, rTest = split_dataset('region_polygon')
        createYAML(rawClasses, regionClasses, 'region_polygon')
        push(rTrain, rTrainImage, rTrainLabel, 'region_polygon', ifJson)
        push(rTest, rTestImage, rTestLabel, 'region_polygon', ifJson)
        push(rValid, rValidImage, rValidLabel, 'region_polygon', ifJson)

if __name__ == '__main__':
    temp = ''
    videos = ['1.m4v', '2.m4v', '3.m4v','4.m4v','5.m4v','6.m4v','7.m4v','8.m4v','9.m4v','10.m4v','11.m4v','12.m4v','13.m4v','14.m4v','15.m4v']
    annotations = ['1.json', '2.json', '3.json','4.json','5.json','6.json','7.json','8.json','9.json','10.json','11.json','12.json','13.json','14.json','15.json']
    """
    print('Press * to stop inputting.')
    while temp != '*':
        temp = input('Example Input (example.mp4)\nVideo Path: ')
        if os.path.exists(temp):
            temp2 = input('Example Input (example.json)\nJson Path: ')
            if os.path.exists(temp2):
                videos.append(temp)
                annotations.append(temp2)
            else:
                if temp2 != '*':
                    print('File not in directory!\n')
        else:
            if temp != '*':
                print('File not in directory!\n')
    """
    v = vidsToYolo(videos, annotations)