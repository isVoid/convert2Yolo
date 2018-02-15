#-*- coding: utf-8 -*-

from xml.etree.ElementTree import Element, dump, ElementTree, parse
import xml.etree.ElementTree as ET
from msgLogInfo import color
from PIL import Image

def VocPascal(converter):



    print(color.BOLD + color.RED + "------------------------- CSV Parsing Start-------------------------" + color.END)

    if label_list is None:
        label_list = self.label_list

    work_dir = getcwd() + "/" + dataSet_dir
    anno_dir = work_dir + anno_dir
    label_dir = work_dir + label_dir

    print("Input file : {}".format(anno_dir))

    f = open(anno_dir, 'r', encoding='utf-8')
    l = csv.reader(f)
    try:
        for line in l:
            print(
                color.BOLD + color.RED + "------------------------- CSV Parsing -------------------------" + color.END)
            convertList = line[0].split(" ")
            length = len(convertList)

            image_name = convertList[0]
            xmin = convertList[1]
            ymin = convertList[2]
            xmax = convertList[3]
            ymax = convertList[4]
            label = convertList[6].split('"')[1]

            if length is 8:
                state = convertList[7].split('"')[1]
                label = label + state

            # Open output result files

            img = Image.open(dataSet_dir + "JPEG/" + image_name)
            img_width = int(img.size[0])
            img_height = int(img.size[1])
            img_depth = 3  # int(img.size[2])

            print("image size (width, height) : {}".format(img.size))
            print()

            print("Output : {}".format(label_dir + image_name[:-3] + "xml"))
            print()

            print("class name, index : ({})".format(label))

            result_outpath = str(label_dir + image_name[:-3] + "xml")

            if not os.path.isfile(result_outpath):
                xml_annotation = Element("annotation")

                xml_folder = Element("folder")
                xml_folder.text = "udacity"

                xml_annotation.append(xml_folder)

                xml_filename = Element("filename")
                xml_filename.text = str(image_name)
                xml_annotation.append(xml_filename)

                xml_path = Element("path")
                xml_path.text = str(label_dir + image_name)
                xml_annotation.append(xml_path)

                xml_source = Element("source")

                xml_database = Element("database")
                xml_database.text = "Unknown"
                xml_source.append(xml_database)
                xml_annotation.append(xml_source)

                xml_size = Element("size")
                xml_width = Element("width")
                xml_width.text = str(img_width)
                xml_size.append(xml_width)

                xml_height = Element("height")
                xml_height.text = str(img_height)
                xml_size.append(xml_height)

                xml_depth = Element("depth")
                xml_depth.text = str(img_depth)
                xml_size.append(xml_depth)

                xml_annotation.append(xml_size)

                xml_segmented = Element("segmented")
                xml_segmented.text = "0"

                xml_annotation.append(xml_segmented)

                xml_object = Element("object")

                xml_name = Element("name")
                xml_name.text = label
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
                xml_xmin.text = str(xmin)
                xml_bndbox.append(xml_xmin)

                xml_ymin = Element("ymin")
                xml_ymin.text = str(ymin)
                xml_bndbox.append(xml_ymin)

                xml_xmax = Element("xmax")
                xml_xmax.text = str(xmax)
                xml_bndbox.append(xml_xmax)

                xml_ymax = Element("ymax")
                xml_ymax.text = str(ymax)
                xml_bndbox.append(xml_ymax)
                xml_object.append(xml_bndbox)

                xml_annotation.append(xml_object)

                self.indent(xml_annotation)
                dump(xml_annotation)
                ElementTree(xml_annotation).write(result_outpath)
            else:
                tree = parse(result_outpath)
                xml_annotation = tree.getroot()
                xml_object = Element("object")

                xml_name = Element("name")
                xml_name.text = label
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
                xml_xmin.text = str(xmin)
                xml_bndbox.append(xml_xmin)

                xml_ymin = Element("ymin")
                xml_ymin.text = str(ymin)
                xml_bndbox.append(xml_ymin)

                xml_xmax = Element("xmax")
                xml_xmax.text = str(xmax)
                xml_bndbox.append(xml_xmax)

                xml_ymax = Element("ymax")
                xml_ymax.text = str(ymax)
                xml_bndbox.append(xml_ymax)
                xml_object.append(xml_bndbox)

                xml_annotation.append(xml_object)
                self.indent(xml_annotation)
                dump(xml_annotation)
                ElementTree(xml_annotation).write(result_outpath)

            print(
                color.BOLD + color.RED + "------------------------- CSV Parsing -------------------------" + color.END)

        print(
            color.BOLD + color.RED + "------------------------- CSV Parsing END -------------------------" + color.END)
    except Exception as e:
        print(color.BOLD + color.RED + "ERROR : {}".format(e) + color.END)