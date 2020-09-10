import os, sys
import numpy as np
import cv2
import time
from lib_inst_det.OID import OID
sys.path.append('lib_inst_det')
from enum import Enum


PROC_LIST = Enum('PROC_LIST', 'Cap Reg Det')
PROC_MODE = PROC_LIST.Det       # default mode is Det(ection)


def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist


# Demo code for object detection and instance classification
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not open. Please check connection.")

    # load the detector ans classifier
    oid = OID('models')     # everything is under this path

    while True:
        ret, frame = cap.read()

        if ret:
            image_np = np.array(frame)  # convert to numpy
            im2show = np.copy(image_np)  # copy for visualization

            if PROC_MODE == PROC_LIST.Cap:
                res_det = []

            elif PROC_MODE == PROC_LIST.Det:
                det_tic = time.time()
                res_det = oid.detect(image_np)
                det_toc = time.time()
                print('Detection took %.3fs for %d object proposals' % (det_toc - det_tic, len(res_det)))

            elif PROC_MODE == PROC_LIST.Reg:
                res_det, num_saved_images = oid.register(image_np, class_name, inst_name)
                print('Registered %s - %s: %d images' % (class_name, inst_name, num_saved_images))

                if num_saved_images == 150:     # capture until this number of images
                    print('Registration is finished')
                    PROC_MODE = PROC_LIST.Det
                    oid.register_finish()   # reload model

            im2show = oid.visualize(im2show, res_det)

            cv2.imshow('im2show', im2show)
            cv2.waitKey(10)
        else:
            print('no frame')

        input_key = cv2.waitKey(10)

        # process input_key
        if input_key == ord('c'):
            print('changing to just capture mode')
            PROC_MODE = PROC_LIST.Cap

        elif input_key == ord('r'):
            list_possible_category = oid.detector.get_possible_class()
            print("Current possible category: ", list_possible_category)
            class_name = input('What category? ex: cup >> ')
            inst_name = input('Who has it? >> ')
            # class_name = 'cup'
            # inst_name = 'w'

            if class_name not in list_possible_category:
                print('we cannot find the object category in the list.')
            else:
                print('changing to registration mode')
                print('registering >> %s - %s' % (class_name, inst_name))
                PROC_MODE = PROC_LIST.Reg
                oid.register_prepare(class_name, inst_name)

        elif input_key == ord('d'):
            PROC_MODE = PROC_LIST.Det

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

