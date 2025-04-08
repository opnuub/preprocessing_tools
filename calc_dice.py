import json
import os
import numpy as np
from glob import glob
from shapely.geometry import Polygon

ROOT_DIR = os.getcwd()
true_dir, pred_dir = os.path.join(ROOT_DIR, 'true'), os.path.join(ROOT_DIR, 'predict')

if __name__ == '__main__':
    files = glob(true_dir + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    dices = []
    for file in files:
        if os.path.exists(os.path.join(pred_dir, f'{file}.json')):
            pred_f = open(f'predict/{file}.json', 'r')
            true_f = open(f'true/{file}.json', 'r')
            pred_file = json.load(pred_f)
            true_file = json.load(true_f)

            for i in pred_file['shapes']:
                for j in true_file['shapes']:
                    try:
                        poly_1 = Polygon(i['points'])
                        poly_2 = Polygon(j['points'])
                        dice = 2 * (poly_1.intersection(poly_2).area) / (poly_1.area + poly_2.area)
                        dices.append(dice)
                    except Exception:
                        pass

            pred_f.close()
            true_f.close()

    final_dice = 0
    dice_count = 0
    for k in dices:
        if k != 0:
            final_dice += k
            dice_count += 1
    missing_count = len(glob(pred_dir + "\\*.jpg")) - len(glob(pred_dir + "\\*.txt"))
    print('Dice:', final_dice/(dice_count+missing_count))
    print('No Detection:', missing_count)