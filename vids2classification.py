from sklearn.model_selection import train_test_split
from glob import glob
import shutil
import json
import cv2
import os

ROOT_DIR = os.getcwd()

def extract_video(filename, suffix):
    input_video = cv2.VideoCapture(f'{filename}.{suffix}')
    f = open(f'{filename}.json')
    json_file = json.load(f)
    w, h = int(input_video.get(3)), int(input_video.get(4))
    size = (w, h)
    fps = json_file['annotation']['video']['fps']
    annotations = json_file['annotation']['actionAnnotationList']
    for annotation in annotations:
        start, end, action_id = annotation["start"], annotation["end"], annotation["action"]
        output_video = cv2.VideoWriter(f'{action_id}_{int(start)}{int(end)}.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, size)
        while start <= end:
            input_video.set(cv2.CAP_PROP_POS_MSEC, start*1000)
            ret, frame = input_video.read()
            if ret == True: 
                output_video.write(frame)
            start += 0.1 # 1/fps
        output_video.release()
    input_video.release()

def create_folders():
    train = os.path.join(ROOT_DIR, 'train')
    if not os.path.exists(train):
        os.makedirs(train)
    valid = os.path.join(ROOT_DIR, 'val')
    if not os.path.exists(valid):
        os.makedirs(valid)
    return train, valid

def process_list(lst):
    unique = []
    for object in lst:
        if object not in unique:
            unique.append(object)
    return unique

def create_labels(videos, labels, group):
    filename = f'kinetics_tiny_{group}_video.txt'
    string = ''
    for video in videos:
        video_path = video.replace("\\", "/").split("/")[-1]
        action = video_path.split("_")[0]
        action_id = labels.index(action)
        string = string + f'{video_path} {action_id}\n'
    with open(filename, 'w') as f:
        f.write(string)


def push(files, destination):
    for file in files:
        shutil.move(file, destination)

def split_dataset(files):
    train_files, val_files = train_test_split(files, test_size=0.1, random_state=55)
    return train_files, val_files

if __name__ == '__main__':
    files = glob(ROOT_DIR + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    for file in files:
        extract_video(file, 'MP4')

    video_files = glob(ROOT_DIR + "\\*.avi")
    video_action_names = [i.replace("\\", "/").split("/")[-1].split(".avi")[0].split('_')[0] for i in video_files]
    unique_action_names = process_list(video_action_names)
    train_video_files, valid_video_files = split_dataset(video_files)
    train_dir, valid_dir = create_folders()
    create_labels(train_video_files, unique_action_names, 'train')
    create_labels(valid_video_files, unique_action_names, 'val')
    push(train_video_files, train_dir)
    push(valid_video_files, valid_dir)
    print(unique_action_names)