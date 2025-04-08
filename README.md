# preprocessing_tools

```calc_dice.py``` \
This script computes dice coefficient between predicted and true polygon shapes across multiple JSON files, accounting robustly for missing detections.

```compare.py``` \
This script visualizes and saves side-by-side comparisons of predicted and ground truth images for evaluation, auto-handling missing masks.

```f1.py``` \
This script provides a comprehensive evaluation pipeline for object detection and instance segmentation tasks using COCO-style annotations. 

```json2yolo.py``` \
This script converts JSON annotations to YOLO format for both rectangle and polygon shapes, then splits and organizes the dataset into train, valid, and optionally test sets, while also generating corresponding data.yaml files for training.

```mergelabels.py``` \
This script splits annotated JSON datasets into bbox/polygon formats, renames classes, converts to YOLO format, and organizes folders.

```preprocess_stereo_original.py``` \
This code reads disparity maps from TIFF files, generates occlusion masks using a stereo matching approach, and saves the resulting images.

```split_video.py``` \
This script splits .avi videos into 100-frame segments using OpenCV, saves them with new filenames, and deletes the original videos.

```test.py``` \
This script converts JSON annotations (polygons/rectangles) to YOLO format, organizes datasets, and creates training folders and data.yaml files.

```txt2json.py``` \
This script converts YOLOv7 .txt annotations to LabelMe .json, processes polygon points for smoothing, and embeds base64 image data for visual labeling.

```vids2classification.py``` \
This script processes video files and annotations, extracts video segments based on action annotations, splits them into training/validation datasets, and creates labels for the dataset.

```vids2yolo.py``` \
This script processes video files and annotations, extracting labeled frames from videos and converting them into YOLO-compatible format. It organizes the data into training, validation, and test sets, and creates required YAML and JSON files for further machine learning tasks.

```xml2txt.py``` \
This script converts XML annotation files (typically in PASCAL VOC format) to YOLO format for object detection tasks. It organizes data into training, validation, and test sets, extracts frames from video files, and generates corresponding .txt files. It also creates a YAML configuration for the dataset, containing class names and paths to images for different sets.

```yolo2obb.py``` \
This script processes JSON annotation files, converts them into YOLO-style .txt files, and splits the dataset into training, validation, and test sets. It organizes images and labels into corresponding directories and generates a YAML configuration for dataset paths and classes.
