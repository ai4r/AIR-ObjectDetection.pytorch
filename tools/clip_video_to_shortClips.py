import os
import sys
import numpy as np
import cv2
import shutil
import xml.etree.ElementTree as ET
import math
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# from lib_inst_det.detector import Detector
from lib_inst_det.detectorAIR15 import DetectorAIR15
from lib_inst_det.detectorAIR23 import DetectorAIR23
from lib_inst_det.OID import OIDv2
sys.path.append('lib_inst_det')


def save_to_xml(path_xml, image_size, list_objs):
    # create xml
    anno = ET.Element('annotation')

    # image info
    filename = ET.SubElement(anno, 'filename')
    filename.text = path_xml.split('/')[-1].split('.')[0]

    size = ET.SubElement(anno, 'size')

    size_w = ET.SubElement(size, 'width')
    size_w.text = str(image_size[0])

    size_h = ET.SubElement(size, 'height')
    size_h.text = str(image_size[1])

    size_c = ET.SubElement(size, 'depth')
    size_c.text = str(image_size[2])

    for item in list_objs:
        classname, xx, yy, xxx, yyy, conf_score = item

        xx = math.floor(float(xx))
        xxx = math.floor(float(xxx))
        yy = math.floor(float(yy))
        yyy = math.floor(float(yyy))

        x1 = min(xx, xxx)
        x2 = max(xx, xxx)
        y1 = min(yy, yyy)
        y2 = max(yy, yyy)

        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0

        obj = ET.SubElement(anno, 'object')

        name = ET.SubElement(obj, 'name')
        name.text = classname

        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        diff = ET.SubElement(obj, 'difficult')
        diff.text = '0'

        score = ET.SubElement(obj, 'score')
        score.text = str(conf_score)

        bndbox = ET.SubElement(obj, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x1)

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(y1)

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x2)

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(y2)

    ET.ElementTree(anno).write(path_xml)

# Clip video to short videos
if __name__ == '__main__':
    from_path = '../test_input_images'
    to_path = '../test_output_images'

    fps = 30

    video_filename = 'Rv_20210621150000-20210621151312_Han.mp4'
    # seq-1
    # start_frame = fps * (2 * 60 + 42)
    # # stop_frame = fps * (4 * 60 + 15)
    # seq-2
    # start_frame = fps * (5 * 60 + 10)
    # # stop_frame = fps * (7 * 60 + 00)
    # seq-3
    start_frame = fps * (8 * 60 + 20)
    stop_frame = fps * (10 * 60 + 10)
    # seq-4
    # start_frame = fps * (10 * 60 + 42)
    # stop_frame = fps * (12 * 60 + 42)

    # video_filename = 'Rv_20210621151426-20210621152229_Han.mp4'
    # start_frame = 0
    # stop_frame = 1000000000000000

    # video_filename = 'R_2021-04-07-14-00-00_2021-04-07-15-00-00.mp4'
    # video_filename = 'R_20210409111433-20210409113512.mp4'

    save_image_anno = True      # JPEGImages, Annotations, ImagesSets
    imshow_result = True


    # start_frame = fps * (3 * 60 + 42)
    # stop_frame = fps * (10 * 60 + 30)
    # start_frame = fps * (4 * 60 + 20)
    # stop_frame = 1000000000000

    plabel_th = 0.7

    # frame rate
    # 30fps
    # 1 img/s -> 30
    # 2 img/s -> 15
    # 5 img/s -> 6
    frame_rate = 6

    from_path_video = os.path.join(from_path, video_filename)
    video_filename_only = os.path.splitext(video_filename)[0]

    path_to_plabel_images = os.path.join(to_path, video_filename_only, 'JPEGImages')
    if not os.path.exists(path_to_plabel_images):
        os.makedirs(path_to_plabel_images)

    path_to_plabel_annos = os.path.join(to_path, video_filename_only, 'Annotations')
    if not os.path.exists(path_to_plabel_annos):
        os.makedirs(path_to_plabel_annos)

    path_to_plabel_imagesets = os.path.join(to_path, video_filename_only, 'ImageSets')
    if not os.path.exists(path_to_plabel_imagesets):
        os.makedirs(path_to_plabel_imagesets)

    path_to_plabel_debug = os.path.join(to_path, video_filename_only, 'Debug')
    if not os.path.exists(path_to_plabel_debug):
        os.makedirs(path_to_plabel_debug)

    # load the detector ans classifier
    # detector = DetectorAIR15('../models', threshold=0.5)  # everything is under this path
    # detector = DetectorAIR23('../models', threshold=0.5)  # everything is under this path
    detector = OIDv2('../models', use_detectorAIR23=True)


    cap = cv2.VideoCapture(from_path_video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if not cap.isOpened():
        raise RuntimeError("Video file could not be opened. Please check the file.", from_path_video)

    output_index = 0
    if save_image_anno:
        fid = open(os.path.join(path_to_plabel_imagesets, 'list.txt'), 'w')
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame_index % frame_rate == 0:
                image_np = np.array(frame)
                im2show = np.copy(image_np)

                image_whc = [image_np.shape[1], image_np.shape[0], image_np.shape[2]]

                res_det = detector.detect(image_np)

                list_objs = []
                for object_index, item in enumerate(res_det):
                    bbox = item[0]
                    score = item[1]
                    class_name = item[2]

                    if score > plabel_th:
                        # if class_name in list_target_object:
                        x1, y1, x2, y2 = bbox

                        list_objs.append([class_name, bbox[0], bbox[1], bbox[2], bbox[3], score])

                if imshow_result:
                    im2show = detector.visualize(im2show, res_det, thresh=plabel_th, fontsize=15)
                    cv2.imshow('im2show', im2show)
                    cv2.waitKey(10)

                if save_image_anno:
                    # save image, anno, filename
                    plabel_filename = 'output-%05d' % output_index # frame_index
                    output_index += 1

                    path_filename_anno = os.path.join(path_to_plabel_annos, plabel_filename + '.xml')
                    path_filename_img = os.path.join(path_to_plabel_images, plabel_filename + '.png')
                    path_filename_debug = os.path.join(path_to_plabel_debug, plabel_filename + '.jpg')

                    save_to_xml(path_filename_anno, image_whc, list_objs)

                    fid.write('%s\n' % plabel_filename)

                    cv2.imwrite(path_filename_img, image_np)

                    cv2.imwrite(path_filename_debug, im2show)


            if frame_index > stop_frame:
                break
        else:
            break

    if save_image_anno:
        fid.close()

    cap.release()
    cv2.destroyAllWindows()

