import json
import os
import numpy as np
from random import randint
from shapely.geometry import Polygon
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def extract_data(filename='instances_default.json'):
        with open(filename, 'r') as f:
                json_file = json.load(f)
                categories = []
                img_filenames = []
                gt = []
                for key, value in json_file.items():
                        if key == 'categories':
                                categories = [{'id': v['id'], 'name': v['name']} for v in value]
                        if key == 'images':
                                img_filenames = [{'image_id': v['id'], 'file_name': v['file_name'], 'width': v['width'], 'height': v['height']} for v in value]
                        if key == 'annotations':
                                for vv in value:
                                        img_id = vv['image_id']
                                        cat_id = vv['category_id']
                                        attributes_raw, attributes = vv['attributes'], []
                                        for k, v in attributes_raw.items():
                                                if 'type' in k and v != '-':
                                                        attributes.append(v)
                                        gt.append({'image_id': img_id, 'label': cat_id, 'items': attributes, 'seg': vv['segmentation'], 'bbox': vv['bbox']})
        return categories, img_filenames, gt

def sorting(dictionary, class_find=False):
        total = []
        newlst = []
        classes = []
        for _ in range(50):
                total.append([])
                classes.append([])
        for i in dictionary:
                idx = i['image_id']
                lst = total[idx-1]
                class_lst = classes[idx-1]
                if class_find:
                        for ii in i['items']:
                                if ii not in class_lst:
                                        class_lst.append(ii)
                lst.append(i)
        for j in total:
                z = sorted(j, key=lambda d: d['items']) 
                newlst.append(z)
        return newlst, classes

def is_seg(gt, pred):
        gtx, predx = gt['seg'], pred['seg']
        if gtx == [[]] or predx == [[]]:
                return False
        else:
                return True

def process(gt, pred, img, classes):
        tp_1, fn_1 = 0, 0
        tp_1_dice, fn_1_dice = 0, 0
        for ii in range(len(gt)):
                tp_2, fn_2 = 0, 0
                tp_2_dice, fn_2_dice = 0, 0
                class_v = classes[ii]
                temp_bbox = [[0, 0] for _ in range(len(class_v))] # tp, fn
                temp_seg = [[0, 0] for _ in range(len(class_v))]
                for jj in range(len(gt[ii])):
                        gtx, prediction = gt[ii][jj], pred[ii][jj]
                        gtx_label, pred_label = gtx['items'], prediction['items']
                        for label in gtx_label:
                                indexx = class_v.index(label)
                                if label in pred_label:
                                        temp_bbox[indexx][0] = temp_bbox[indexx][0] + 1
                                        if is_seg(gtx, prediction):
                                                temp_seg[indexx][0] = temp_seg[indexx][0] + 1
                                                tp_2_dice += 1
                                        tp_2 += 1
                                else:
                                        temp_bbox[indexx][1] = temp_bbox[indexx][1] + 1
                                        if is_seg(gtx, prediction):
                                                temp_seg[indexx][1] = temp_seg[indexx][1] + 1
                                                fn_2_dice += 1
                                        fn_2 += 1
                tp_1 += tp_2
                fn_1 += fn_2
                tp_1_dice += tp_2_dice
                fn_1_dice += fn_2_dice
                filenamee = img[ii]["file_name"].split('/')[-1].split('.')[0]
                with open(f'folder/{filenamee}.txt', 'w') as f:
                        string_bbox = 'bbox f1-score: '
                        string_seg = 'segmentation dice: '
                        for xx in range(len(temp_bbox)):
                                bbox_set, seg_set = temp_bbox[xx], temp_seg[xx]
                                try:
                                        f12 = (2 * bbox_set[0]) / (2 * bbox_set[0] + bbox_set[1])
                                except Exception:
                                        f12 = 0
                                try:
                                        dice2 = (2 * seg_set[0]) / (2 * seg_set[0] + seg_set[1])
                                except Exception:
                                        dice2 = 0
                                item = class_v[xx]
                                string_bbox = string_bbox + f'{item} {f12:.2f},'
                                string_seg = string_seg + f'{item} {dice2:.2f},'
                        try:
                                f1 = (2 * tp_2) / (2 * tp_2 + fn_2)
                        except Exception:
                                f1 = 0
                        try:
                                dice = (2 * tp_2_dice) / (2 * tp_2_dice + fn_2_dice)
                        except Exception:
                                dice = 0
                        full = f"{string_bbox.strip(',')}\n{string_seg.strip(',')}\n\nOverall F1-Score: {f1:.2f}\nOverall Dice: {dice:.2f}"
                        f.write(full)
        try:
                f11 = (2 * tp_1) / (2 * tp_1 + fn_1)
        except Exception:
                f11 = 0
        try:
                dice1 = (2 * tp_1_dice) / (2 * tp_1_dice + fn_1_dice)
        except Exception:
                dice1 = 0
        print(f11)
        print(dice1)

def get_label(gt):
        lst = []
        for x in gt:
                label = x['label']
                if label not in lst:
                        lst.append(label)
        return lst

def preprocess(gt, pred):
        gtx_return, predx_return = [], []
        for ii in range(len(gt)):
                for jj in range(len(gt[ii])):
                        gtx, predx = gt[ii][jj], pred[ii][jj]
                        gtx_seg, predx_seg = gtx['seg'], predx['seg']
                        if len(gtx_seg)!=0: 
                                gtx_seg = gtx_seg[0]
                                gtx_new_seg = []
                                double = []
                                for x in range(1, len(gtx_seg)+1):
                                        pts = gtx_seg[x-1]
                                        if x%2!=0:
                                                double.append(pts)
                                        else:
                                                double.append(pts)
                                                gtx_new_seg.append(double)
                                                double = []
                                gtx_return.append(gtx_new_seg)
                        if len(predx_seg)!=0: 
                                predx_seg = predx_seg[0]
                                predx_new_seg = []
                                double = []
                                for x in range(1, len(predx_seg)+1):
                                        pts = predx_seg[x-1]
                                        if x%2!=0:
                                                double.append(pts)
                                        else:
                                                double.append(pts)
                                                predx_new_seg.append(double)
                                                double = []
                                predx_return.append(predx_new_seg)
        return gtx_return, predx_return

def process_seg_iou(gt_seg, pred_seg):
        total_dice = 0
        count = 0
        for ii in range(len(gt_seg)):
                seg_gt, seg_pred = gt_seg[ii], pred_seg[ii]
                if len(seg_gt) > 0:
                        try:
                                count += 1
                                poly_1 = Polygon(seg_gt)
                                poly_2 = Polygon(seg_pred)
                                dice = (poly_1.intersection(poly_2).area) / (poly_1.area + poly_2.area)
                                total_dice += dice
                        except Exception:
                                pass
        return total_dice/count
        #score = f1_score(tp, fp, fn)
        #poly_2 = Polygon(segg['seg'])
        #iou = (poly_2).area

def f1_score(tp, fp, fn): # Also dice for segmentation
        return 2 * tp / (2 * tp + fp + fn)

def confusion_metrics(gt, pred):
        gt_labels, pred_labels = [], []
        for ii in range(len(gt)):
                for jj in range(len(gt[ii])):
                        gtx, predx = gt[ii][jj], pred[ii][jj]
                        gt_labels.append(gtx['label'])
                        pred_labels.append(predx['label'])
        y_true, y_pred = np.array(gt_labels), np.array(pred_labels)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(cm, cmap=plt.cm.Blues) # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
        fig.colorbar(cax)
        ax.set(title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=np.arange(2),
        yticklabels=np.arange(2))
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.title.set_size(20)
        threshold = (cm.max() + cm.min()) / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                        horizontalalignment="center",
                        color="white" if cm[i, j] > threshold else "black",
                        size=15)
        plt.savefig('folder/confusion_matrix.png')

if __name__ == '__main__':
        threshold = 0.95

        ROOT_DIR = os.getcwd()
        dir = os.path.join(ROOT_DIR, 'folder')
        if not os.path.exists(dir):
                os.makedirs(dir)

        #Let 1 be true and 2 be false

        categories, img_filenames, gt = extract_data('9_gt.json')
        _, _, pred = extract_data('9_pred.json')

        gt, class_list = sorting(gt, True)  # Structure: 1, 50, idk
        pred, _ = sorting(pred, False)

        """
        gtx_seg, predx_seg = preprocess(gt, pred)
        dice = process_seg_iou(gtx_seg, predx_seg)
        print(dice)
        """
        
        process(gt, pred, img_filenames, class_list)

        # confusion_metrics(gt, pred)

        """
        Get the classes for each image_id (total 50)
        Make a list of each classes' tp, fp, fn
        * Make if_seg function to differentiate dice
        for class in classes, write to the txt file
        
        filenamee = img[ii]["file_name"].split('/')[-1].split('.')[0]
                with open(f'folder/{filenamee}.txt', 'a') as f:
                        try:
                                f1 = (2 * ttp) / (2 * ttp + ffp + ffn)
                        except Exception:
                                f1 = 0
                        f.write(f'\nsegmentation Dice: {f1}')
        """