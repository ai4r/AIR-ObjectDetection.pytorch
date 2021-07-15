import _init_paths
import os, sys
import numpy as np
import cv2
import time
# from lib_inst_det.OID import OID
from lib_inst_det.OID import OIDv2	# include AIR-15 detector and COCO detector
from lib_inst_det.detectorAIR23 import DetectorAIR23
import json	# to save detection results in files
sys.path.append('lib_inst_det')

# use only GPU-1
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    # cap = cv2.VideoCapture(1)     # video input
    cap = cv2.VideoCapture('test_input_images/free%02d.jpg')        # image sequence input

    do_flip_input = False	# if you want to apply flipLR to input image

    # resizing resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    SAVE_INPUT_IMAGES = False		# input images will be stored in the 'path_to_save_images' folder
    SAVE_OUTPUT_IMAGES = True       # out images will be stored in the 'path_to_save_images' folder

    th_vis = 0.8
    model_filename = 'faster_rcnn_1_10_9999_syncloud_public23_cloud_testbedv2x6_FPN_CBAM.pth'

    path_to_save_images = os.path.join('test_output_images', model_filename)

    if not os.path.exists(path_to_save_images):
        os.makedirs(path_to_save_images)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not open. Please check connection.")

    # load the detector ans classifier
    # oid = OID('models')     # everything is under this path
    # oid = OIDv2('models')     # everything is under this path
    oid = DetectorAIR23(baseFolder='models',
                        filename=model_filename,
                        threshold=th_vis,
                        att_type='CBAM'
                        )  # everything is under this path

    count = 0
    while True:
        ret, frame = cap.read()     # bgr
    
        if do_flip_input:
            frame = cv2.flip(frame, 1)

        if ret:
            image_np = np.array(frame)  # convert to numpy
            im2show = np.copy(image_np)  # copy for visualization

            det_tic = time.time()
            res_det = oid.detect(image_np)
            det_toc = time.time()
            print('Detection took %.3fs for %d object proposals' % (det_toc - det_tic, len(res_det)))

            im2show = oid.visualize(im2show, res_det, thresh=th_vis, map_classname_to_korean=oid.display_classes)

            cv2.imshow('im2show', im2show)
            cv2.waitKey(10)

            if SAVE_INPUT_IMAGES:
                cv2.imwrite(os.path.join(path_to_save_images, 'input_%05d.png' % count), image_np)

            if SAVE_OUTPUT_IMAGES:
                cv2.imwrite(os.path.join(path_to_save_images, 'output_%05d.png' % count), im2show)
                # with open(os.path.join(path_to_save_images, 'result_%5d.json'), 'w') as fp:
                #     json.dump(res_det, fp)

                count = count + 1
        else:
            print('no frame')

        input_key = cv2.waitKey(10)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

