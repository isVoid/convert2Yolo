#-*- coding: utf-8 -*-

import json
import os
from pprint import pprint
import cv2
import time
from PIL import Image
from xml.etree.ElementTree import Element, dump, ElementTree, parse
import xml.etree.ElementTree as ET

from msgLogInfo import color
from convert2Yolo import bcolors, Logger, get_file_list

class COCO_INFO :
    annotation = {'1'    :   'person',
                  '2'    :   'bicycle',
                  '3'    :   'car',
                  '4'    :   'motorcycle',
                  '5'    :   'airplane',
                  '6'    :   'bus',
                  '7'    :   'train',
                  '8'    :   'truck',
                  '9'    :   'boat',
                  '10'   :   'traffic_light',
                  '11'   :   'fire_hydrant',
                  '12'   :   'unknown',
                  '13'   :   'stop_sign',
                  '14'   :   'parking_meter',
                  '15'   :   'bench',
                  '16'   :   'bird',
                  '17'   :   'cat',
                  '18'   :   'dog',
                  '19'   :   'horse',
                  '20'   :   'sheep',
                  '21'   :   'cow',
                  '22'   :   'elephant',
                  '23'   :   'bear',
                  '24'   :   'zebra',
                  '25'   :   'giraffe',
                  '26'   :   'unknown',
                  '27'   :   'backpack',
                  '28'   :   'umbrella',
                  '29'   :   'unknown',
                  '30'   :   'unknown',
                  '31'   :   'handbag',
                  '32'   :   'tie',
                  '33'   :   'suitcase',
                  '34'   :   'frisbee',
                  '35'   :   'skis',
                  '36'   :   'snowboard',
                  '37'   :   'sports_ball',
                  '38'   :   'kite',
                  '39'   :   'baseball_bat',
                  '40'   :   'baseball_glove',
                  '41'   :   'skateboard',
                  '42'   :   'surfboard',
                  '43'   :   'tennis_racket',
                  '44'   :   'bottle',
                  '45'   :   'unknown',
                  '46'   :   'wine_glass',
                  '47'   :   'cup',
                  '48'   :   'fork',
                  '49'   :   'knife',
                  '50'   :   'spoon',
                  '51'   :   'bowl',
                  '52'   :   'banana',
                  '53'   :   'apple',
                  '54'   :   'sandwich',
                  '55'   :   'orange',
                  '56'   :   'broccoli',
                  '57'   :   'carrot',
                  '58'   :   'hot_dog',
                  '59'   :   'pizza',
                  '60'   :   'donut',
                  '61'   :   'cake',
                  '62'   :   'chair',
                  '63'   :   'couch',
                  '64'   :   'potted_plant',
                  '65'   :   'bed',
                  '66'   :   'unknown',
                  '67'   :   'dining_table',
                  '68'   :   'unknown',
                  '69'   :   'unknown',
                  '70'   :   'toilet',
                  '71'   :   'unknown',
                  '72'   :   'tv',
                  '73'   :   'laptop',
                  '74'   :   'mouse',
                  '75'   :   'remote',
                  '76'   :   'keyboard',
                  '77'   :   'cell_phone',
                  '78'   :   'microwave',
                  '79'   :   'oven',
                  '80'   :   'toaster',
                  '81'   :   'sink',
                  '82'   :   'refrigerator',
                  '83'   :   'unknown',
                  '84'   :   'book',
                  '85'   :   'clock',
                  '86'   :   'vase',
                  '87'   :   'scissors',
                  '88'   :   'teddy_bear',
                  '89'   :   'hair_drier',
                  '90'   :   'toothbrush',
                   }

font = cv2.FONT_HERSHEY_SIMPLEX

def COCOtoVoc(converter):

    anno_dir = converter.anno_dir
    image_dir = converter.image_dir
    output_dir = converter.output_dir
    classes = converter.classes
    image_type = converter.image_type
    manifest_dir = converter.manifest_dir
    class_list = COCO_INFO.annotation
    indent = converter.indent

    logger = Logger(mode = 'save', output_dir = manifest_dir)

    ##########################################
    # Open Json file                         #
    ##########################################
    # '/media/keti-1080ti/ketiCar/handling_DataSet/COCO/annotations/instances_train2014.json'
    json_data = json.load(open(anno_dir))

    pre_meta = image_dir.split("/")
    len_pre_meta = len(pre_meta)
    file_meta = pre_meta[len_pre_meta - 3] + "_" + pre_meta[len_pre_meta - 2] + "_"


    ##########################################
    # Image List                             #
    ##########################################
    image_list = []

    while True:

        count = 0
        for anno in json_data["annotations"]:

            ##########################################
            # Overwrapping Image Check               #
            ##########################################
            if len(image_list) == 0:
                flag = False
            else:
                for i in image_list:
                    if (anno["image_id"] == i):
                        flag = True
                    else:
                        flag = False


            ##########################################
            # Only non-overwrapping Image            #
            ##########################################
            if flag == False:


                ##########################################
                # Dict for multi object                  #
                ##########################################
                prop = {}
                obj = {}


                ##########################################
                # Add overwraped Image                   #
                ##########################################
                image_list.append(anno["image_id"])

                obj_cnt = 1


                ##########################################
                # Annotation Parsing                     #
                ##########################################
                image_id = str(anno["image_id"]).zfill(12)
                image_name = file_meta + image_id + ".jpg"
                image_file = image_dir + image_name
                cls_idx = str(anno['category_id'])
                cls = str(class_list[cls_idx])
                box = [int(anno["bbox"][0]), int(anno["bbox"][1]), int(anno["bbox"][2]), int(anno["bbox"][3])]

                obj["name"] = cls
                obj["box"] = box

                prop["object" + str(obj_cnt)] = obj

                ##########################################
                # Search Same Image and different label  #
                ##########################################
                for aux_anno in json_data["annotations"]:

                    ##########################################
                    # Annotation Parsing                     #
                    ##########################################
                    aux_cls_idx = str(aux_anno['category_id'])
                    aux_cls = str(class_list[aux_cls_idx])
                    aux_box = [int(aux_anno["bbox"][0]), int(aux_anno["bbox"][1]), int(aux_anno["bbox"][2]),
                               int(aux_anno["bbox"][3])]


                    ##########################################
                    # Validate overwrap annotation           #
                    ##########################################
                    x_flag = (int(anno["bbox"][0]) == aux_box[0])
                    y_flag = (int(anno["bbox"][1]) == aux_box[1])
                    width_flag = (int(anno["bbox"][2]) == aux_box[2])
                    height_flag = (int(anno["bbox"][3]) == aux_box[3])

                    coord_flag = ((x_flag == True) and (y_flag == True) and (width_flag == True) and (
                                height_flag == True))

                    if (len(image_list) == 0):
                        image_same_flag = False

                    for i in image_list:
                        if (aux_anno["image_id"] == i):
                            image_same_flag = True
                            # print("origin : {}, compare :{}".format(str(anno["image_id"]) ,str(aux_anno["image_id"])))
                            # print("coord_flag : {}".format(coord_flag))
                            # print("o_box : {}, c_box : {}".format(anno["bbox"], aux_anno["bbox"]))
                            # print("o_box : {}, c_box : {}".format(box, aux_box))

                        else:
                            image_same_flag = False

                    ##########################################
                    # Add multi label                        #
                    ##########################################
                    if (image_same_flag == True):
                        if (coord_flag == False):
                            aux_obj = {}

                            obj_cnt += 1

                            aux_obj["name"] = aux_cls
                            aux_obj["box"] = aux_box
                            prop["object" + str(obj_cnt)] = aux_obj

            print(prop)

            ##########################################
            # Image open and get size                #
            ##########################################
            print("Image file path : {}".format(image_file))
            print()
            image = Image.open(image_file)
            image_width = int(image.size[0])
            image_height = int(image.size[1])
            image_detph = 3

            print("Image Size (width, height) : {}".format(image.size))
            print()


            ##########################################
            # Make xml file                          #
            ##########################################
            output = output_dir + image_name[:-3] + "xml"

            print("Output : {}".format(output))
            print()

            if not os.path.isfile(output):
                xml_annotation = Element("annotation")

                xml_folder = Element("filder")
                xml_folder.text = "coco"

                xml_annotation.append(xml_folder)

                xml_filename = Element("filename")
                xml_filename.text = str(image_name)
                xml_annotation.append(xml_filename)

                xml_path = Element("path")
                xml_path.text = str(image_file)
                xml_annotation.append(xml_path)

                xml_source = Element("source")

                xml_database = Element("database")
                xml_database.text = "Unknown"
                xml_source.append(xml_database)
                xml_annotation.append(xml_source)

                xml_size = Element("size")
                xml_width = Element("width")
                xml_width.text = str(image_width)
                xml_size.append(xml_width)

                xml_height = Element("height")
                xml_height.text = str(image_height)
                xml_size.append(xml_height)

                xml_depth = Element("depth")
                xml_depth.text = str(image_detph)
                xml_size.append(xml_depth)

                xml_annotation.append(xml_size)

                xml_segmented = Element("segmented")
                xml_segmented.text = "0"

                xml_annotation.append(xml_segmented)

                for _obj in prop.keys():
                    _name = prop[_obj]["name"]
                    _box = prop[_obj]["box"]

                    _xmin = _box[0]
                    _ymin = _box[1]
                    _xmax = _box[0]+_box[2]
                    _ymax = _box[1]+_box[3]

                    print("class name : {}".format(_name))
                    xml_object = Element("object")

                    xml_name = Element("name")
                    xml_name.text = _name
                    xml_object.append(xml_name)

                    xml_pose = Element("pose")
                    xml_pose.text = "Unspecified"
                    xml_object.append(xml_pose)

                    xml_truncated = Element("truncated")
                    xml_truncated.text = "0"
                    xml_object.append(xml_truncated)

                    xml_difficult = Element("difficult")
                    xml_difficult.text = "0"
                    xml_object.append(xml_difficult)

                    xml_bndbox = Element("bndbox")
                    xml_xmin = Element("xmin")
                    xml_xmin.text = str(_xmin)
                    xml_bndbox.append(xml_xmin)

                    xml_ymin = Element("ymin")
                    xml_ymin.text = str(_ymin)
                    xml_bndbox.append(xml_ymin)

                    xml_xmax = Element("xmax")
                    xml_xmax.text = str(_xmax)
                    xml_bndbox.append(xml_xmax)

                    xml_ymax = Element("ymax")
                    xml_ymax.text = str(_ymax)
                    xml_bndbox.append(xml_ymax)
                    xml_object.append(xml_bndbox)

                    xml_annotation.append(xml_object)

                indent(xml_annotation)
                dump(xml_annotation)
                ElementTree(xml_annotation).write(output)