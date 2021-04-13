# coding=utf-8
__author__ = 'yochin'
__version__ = '1.0'

import os
import sys
import cv2
import codecs
import copy
import random as rnd
import numpy as np
import xml.etree.ElementTree as ET
import itertools
from collections import namedtuple
import os.path as osp
from model.utils.net_utils import vis_detections_korean_ext2
import pdb


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def get_overlap_ratio(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    # dx and dy is width and height of IoU

    if (dx>=0) and (dy>=0):
        return float(dx*dy) / float((a.xmax-a.xmin)*(a.ymax-a.ymin) + (b.xmax-b.xmin)*(b.ymax-b.ymin) - dx*dy)
    else:
        return 0.


if __name__ == '__main__':
    # plabel
    saveBaseFolder = '../test_input_images/R_2021-04-07-14-00-00_2021-04-07-15-00-00'
    savedFolder_Infos = osp.join(saveBaseFolder, 'Annotations')
    savedFolder_Images = osp.join(saveBaseFolder, 'JPEGImages')
    savedFolder_ImageSets = osp.join(saveBaseFolder, 'ImageSets')  # If none, then get all files from Images folder and make lists
    listname = 'list.txt'
    image_ext = 'png'

    save_GT_ImagePath = osp.join(saveBaseFolder, 'GTedImages', listname.split('.')[0])

    # Options
    do_save_gt = True
    do_shuffle = False
    do_show_image = False    # show and stop for keying enter
    do_check_outofbound = False
    do_bndbox = True
    xml_zerobase = True
    # True: bbox should range 0 ~ Width-1
    # False: bbox should range 1 ~ Width
    # Faster-RCNN_TF uses 0-based index.
    # PASCAL Annotation uses 1-based index.
    find_small_bbox = False
    # maxmin_small_box = [50, 30]
    maxmin_small_box = [5, 5]

    if do_save_gt:
        if not os.path.exists(save_GT_ImagePath):
            os.makedirs(save_GT_ImagePath)

    # Load file list from text file or data folder
    if savedFolder_ImageSets is not None:
        listFiles = codecs.open(osp.join(savedFolder_ImageSets, listname)).read().split('\n')
    else:
        listFilesExt = list_files(savedFolder_Images, image_ext)
        listFiles = [os.path.splitext(item)[0] for item in listFilesExt]
        print('we generate list from Images folder.')

    # remove item if the last is blank
    if len(listFiles[-1]) == 0:
        del listFiles[-1]

    # check blank space
    for ifile, filename in enumerate(listFiles):
        if len(filename) == 0:
            raise AssertionError('%s-th line is blank'%ifile)

    if do_shuffle == True:
        rnd.shuffle(listFiles)

    list_class = []
    list_sum_bbox = []

    # listFiles = ['3395592']
    # display_class = ['car']

    start_index = 0
    for ifile, filename in enumerate(listFiles[start_index:], start=start_index):
        if ifile%100 == 0:
            print('%d/%d'%(ifile, len(listFiles)))

        if len(filename) > 0:
            strFullPathImage = savedFolder_Images + '/' + filename + '.png'

            if os.path.isfile(strFullPathImage) is False:
                strFullPathImage = savedFolder_Images + '/' + filename + '.jpg'

            if (do_show_image or do_save_gt) or do_check_outofbound:
                im = cv2.imread(strFullPathImage)

            # xml file type
            strFullPathAnno = savedFolder_Infos + '/' + filename + '.xml'
            tree = ET.parse(strFullPathAnno)
            objs = tree.findall('object')
            num_objs = len(objs)  # number of object in one image

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                tx = int(bbox.find('xmin').text)
                ty = int(bbox.find('ymin').text)
                tx2 = int(bbox.find('xmax').text)
                ty2 = int(bbox.find('ymax').text)

                score = float(obj.find('score').text)

                if (do_show_image or do_save_gt) and do_bndbox:
                    # cv2.rectangle(im, (tx, ty), (tx2, ty2), (255, 0, 0), 2)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(im, obj.find('name').text, (tx, ty), font, 1, (255, 255, 255), 2)

                    # if obj.find('name').text in display_class:
                    im = vis_detections_korean_ext2(im, obj.find('name').text, np.asarray([[tx, ty, tx2, ty2, score]]), thresh=0.5)

                if do_check_outofbound == True:
                    # print('%d: %s file - %d object'%(ifile, filename, ix))

                    if xml_zerobase == False:
                        tx = tx-1
                        ty = ty-1
                        tx2 = tx2-1
                        ty2 = ty2-1

                    assert tx >= 0, '%s: left point tx(%d) >= 0' % (filename, tx)
                    assert ty >= 0, '%s: top point ty(%d) >= 0' % (filename, ty)
                    assert tx2 <= im.shape[1]-1, '%s, right point tx2(%d) <= width(%d)' % (filename, tx2, im.shape[1])
                    assert ty2 <= im.shape[0]-1, '%s, bottom point ty2(%d) <= height(%d)' % (filename, ty2, im.shape[0])

                    assert tx < tx2, filename + ', left point tx < right point tx2'
                    assert ty < ty2, filename + ', top point ty < bottom point ty2'

                    # assert float(im.shape[0])/float(im.shape[1]) < 4, '%s, ratio > 4'%(filename)
                    # assert float(im.shape[1])/float(im.shape[0]) < 4, '%s, ratio > 4'%(filename)

                    if np.isnan(tx * ty * tx2 * ty2) is True:
                        print('nan data: %d %d %d %d' % (tx, ty, tx2, ty2))

                # If you want to find certain condition, then list.
                if find_small_bbox == True:
                    box_width = tx2-tx+1
                    box_height = ty2-ty+1

                    if max(box_width, box_height) <= max(maxmin_small_box) or min(box_width, box_height) <= min(maxmin_small_box):
                        print(filename)
                        print('Size (%d, %d) is less than 50x30'%(tx2-tx, ty2-ty))
                        print(obj.find('difficult').text)

                #
                if obj.find('name').text not in list_class:
                    list_class.append(obj.find('name').text)
                    list_sum_bbox.append(0)

                index_class = list_class.index(obj.find('name').text)
                list_sum_bbox[index_class] += 1



            if do_show_image == True:
                print(filename)
                cv2.imshow('img_anno', im)
                cv2.waitKey(0)

            if do_save_gt:
                cv2.imwrite(os.path.join(save_GT_ImagePath, filename + '.jpg'), im)


    for ith, class_name in enumerate(list_class):
        print('%s : %d' % (list_class[ith], list_sum_bbox[ith]))