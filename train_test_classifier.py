import os
import sys
import numpy as np
import cv2
import shutil

from lib_inst_det.detector import Detector
from lib_inst_det.OID import OID
sys.path.append('lib_inst_det')

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def list_dirs(path):
    filelist = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            filelist.append(name)
    return filelist

if __name__ == '__main__':
    path_train = 'AIR_5yr/cropped'
    path_test = 'AIR_5yr/test'

    list_target_object = ['cup', 'remote']  # object in this list is saved as files

    list_classifier = []

    for obj_name in list_target_object:
        path_to_obj = os.path.join(path_train, obj_name)

        list_owners = list_dirs(path_to_obj)


        for owner_name in list_owners:
            path_to_inst = os.path.join(path_train, obj_name, owner_name)

            list_filename = list_files(path_to_inst, 'png')

            for filename in list_filename:
                path_to_image = os.path.join(path_train, obj_name, owner_name, filename)

                im = cv2.imread(path_to_image)
                cv2.imshow('im', im)
                cv2.waitKey(100)

                # pre-processing



                # feature extractor




    # load the detector and classifier
    detector = Detector('models', threshold=0.5)  # everything is under this path

    fid = open('missing_file.txt', 'w')


    # read all cropped images



    # train w/ GMM, kNN, NN_4th



    # test with detector
