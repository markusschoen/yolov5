#-*-coding:utf-8-*-

import os
import json
import tempfile

from tqdm import tqdm
from data_format import COCO, YOLO

attr_dict = dict()
attr_dict["categories"] = [
    {"supercategory": "none", "id":  0, "name": "pedestrian"},
    {"supercategory": "none", "id":  1, "name": "rider"},
    {"supercategory": "none", "id":  2, "name": "car"},
    {"supercategory": "none", "id":  3, "name": "bus"},
    {"supercategory": "none", "id":  4, "name": "truck"},
    {"supercategory": "none", "id":  5, "name": "bicycle"},
    {"supercategory": "none", "id":  6, "name": "motorcycle"},
    {"supercategory": "none", "id":  7, "name": "tl_green"},
    {"supercategory": "none", "id":  8, "name": "tl_red"},
    {"supercategory": "none", "id":  9, "name": "tl_yellow"},
    {"supercategory": "none", "id": 10, "name": "tl_none"},
    {"supercategory": "none", "id": 11, "name": "traffic sign"},
    {"supercategory": "none", "id": 12, "name": "train"}
]

attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

tl_color_map = {
    "R": "red",
    "Y": "yellow",
    "G": "green",
    "NA": "none",
}


def bdd2coco_detection(id_dict, labeled_images, fn):

    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True

        if 'labels' in i:
            for label in i['labels']:
                annotation = dict()
                category=label['category']
                if (category == "traffic light"):
                    color = tl_color_map[label['attributes']['trafficLightColor']]
                    category = "tl_" + color
                if category in id_dict.keys():
                    empty_image = False
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image['id']
                    x1 = label['box2d']['x1']
                    y1 = label['box2d']['y1']
                    x2 = label['box2d']['x2']
                    y2 = label['box2d']['y2']
                    annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
                    annotation['area'] = float((x2 - x1) * (y2 - y1))
                    annotation['category_id'] = id_dict[category]
                    annotation['ignore'] = 0
                    annotation['id'] = label['id']
                    annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    json.dump(attr_dict, fn)


def main(config):
    if config["datasets"] == "COCO":
        coco = COCO()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = coco.parse(config["label"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                        config["img_type"], config["manipast_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("COCO Parsing Result : {}, msg : {}".format(flag, data))

    else:
        print("Unkwon Datasets")


if __name__ == '__main__':
    label_dir = "../../../datasets/bdd_yolo/labels/det_20"

    print('Loading validation set...')
    # create BDD validation set detections in COCO format
    with open(os.path.join(label_dir,
                           'det_val.json')) as f:
        val_labels = json.load(f)
    print('Converting validation set to COCO format...')

    tfile = tempfile.NamedTemporaryFile(mode="w+")
    bdd2coco_detection(attr_id_dict, val_labels, tfile)
    tfile.flush()

    config = {
        "datasets": "COCO",
        "img_path": "../../../datasets/bdd_yolo/images/val",
        "label": tfile.name,
        "img_type": ".jpg",
        "manipast_path": "../../../datasets/bdd_yolo/val.txt",
        "output_path": "../../../datasets/bdd_yolo/labels/val",
        "cls_list": "../bdd_classes.names",
    }

    print('Converting validation set to YOLO format...')
    main(config)

    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(label_dir,
                           'det_train.json')) as f:
        train_labels = json.load(f)
    print('Converting training set to COCO format...')

    tfile = tempfile.NamedTemporaryFile(mode="w+")
    bdd2coco_detection(attr_id_dict, train_labels, tfile)
    tfile.flush()

    config = {
        "datasets": "COCO",
        "img_path": "../../../datasets/bdd_yolo/images/train",
        "label": tfile.name,
        "img_type": ".jpg",
        "manipast_path": "../../../datasets/bdd_yolo/train.txt",
        "output_path": "../../../datasets/bdd_yolo/labels/train",
        "cls_list": "../bdd_classes.names",
    }
    print('Converting training set to YOLO format...')
    main(config)
