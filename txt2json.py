import json
import os
from glob import glob
import base64
import labelme

ROOT_DIR = os.getcwd()

if __name__ == '__main__': # FOR YOLOV7.TXT TO LABELME.JSON

    WIDTH, HEIGHT = 1920, 1080
    labels = {'0': "test0", '1': "test1", '2': "test2"}

    files = glob(ROOT_DIR + "\\*.txt")
    files = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in files]
    
    for file in files:
        shapes = []
        with open(f'{file}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip('\n').split(' ')
                label = labels[info[0]]
                seg_points = []
                for i in range(1, len(info)-1, 2):
                    seg_points.append([float(info[i])*WIDTH, float(info[i+1])*HEIGHT])

                # processing
                NUM=25
                seg_pointss=[seg_points[0]]
                for i in range(len(seg_points)):
                    if abs(seg_points[i][0]-seg_pointss[-1][0]) > NUM or abs(seg_points[i][1]-seg_pointss[-1][1]) > NUM:
                        seg_pointss.append(seg_points[i])

                print(len(seg_pointss))

                shapes.append({
                    "label": label,
                    "points": seg_pointss,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        image_path = f'{file}.jpg'
        data = labelme.LabelFile.load_image_file(image_path)
        image_data = base64.b64encode(data).decode('utf-8')
        dictionary = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": image_data,
            "imageHeight": HEIGHT,
            "imageWidth": WIDTH
        }
        with open(f'{file}.json', 'w') as output:
            json.dump(dictionary, output)
            