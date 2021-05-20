import _init_paths
import os, sys
import numpy as np
import cv2
import time
from lib_inst_det.OID import OID
from lib_inst_det.OID import OIDv2	# include AIR-15 detector and COCO detector
import json	# to save detection results in files
sys.path.append('lib_inst_det')
from enum import Enum


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# updated: add detectorAIR15 for 15 handheld objects
# updated: add save input and output images
#     control capture size

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
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture('images/image.jpg')
    # cap = cv2.VideoCapture('test_input_images/input_1/input_%02d.jpg')
    # cap = cv2.VideoCapture('test_input_images/input_5/%05d.jpg')

    NUM_REG_IMAGES = 50

    do_flip_input = True	# if you want to apply flipLR to input image

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    SAVE_IMAGES = False		# input and out images will be save in the 'path_to_save_images' folder

    path_to_save_images = 'test_output_images'

    if not os.path.exists(path_to_save_images):
        os.makedirs(path_to_save_images)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not open. Please check connection.")

    # load the detector ans classifier
    # oid = OID('models')     # everything is under this path
    oid = OIDv2('models')     # everything is under this path

    count = 0
    while True:
        ret, frame = cap.read()     # bgr
    
        if do_flip_input:
            frame = cv2.flip(frame, 1)

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

                comment = oid.generate_comment([10, 10, 6000, 3000], res_det, 4)
                if not None:
                    print(comment)

            elif PROC_MODE == PROC_LIST.Reg:
                res_det, num_saved_images = oid.register(image_np, class_name, inst_name)
                print('Registered %s - %s: %d images' % (class_name, inst_name, num_saved_images))

                if num_saved_images == NUM_REG_IMAGES:     # capture until this number of images
                    print('Registration is finished')
                    PROC_MODE = PROC_LIST.Det
                    oid.register_finish()   # reload model

            im2show = oid.visualize(im2show, res_det, thresh=0.7, map_classname_to_korean=oid.map_classname_to_korean)

            cv2.imshow('im2show', im2show)
            cv2.waitKey(10)

            if SAVE_IMAGES:
                cv2.imwrite(os.path.join(path_to_save_images, 'input_%05d.png' % count), image_np)
                cv2.imwrite(os.path.join(path_to_save_images, 'output_%05d.png' % count), im2show)
                # with open(os.path.join(path_to_save_images, 'result_%5d.json'), 'w') as fp:
                #     json.dump(res_det, fp)

                count = count + 1
        else:
            print('no frame')

        input_key = cv2.waitKey(10)

        # process input_key
        if input_key == ord('c'):
            print('changing to just capture mode')
            PROC_MODE = PROC_LIST.Cap

        elif input_key == ord('r'):
            # list_possible_category = oid.detector.get_possible_class()
            list_possible_category = oid.detectorAIR15.get_possible_class()
            print("Current possible category: ", list_possible_category)
            class_name = input('What category? ex: cup >> ')
            inst_name = input('Who has it? >> ')
            # class_name = 'cup'
            # inst_name = 'w'

            if class_name not in list_possible_category:
                print('we cannot find the object category in the list.')
            else:
                print('changing to registration mode')
                print('possible object list: ', list_possible_category)
                print('registering >> %s - %s' % (class_name, inst_name))
                PROC_MODE = PROC_LIST.Reg
                oid.register_prepare(class_name, inst_name)

        elif input_key == ord('d'):
            PROC_MODE = PROC_LIST.Det

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

