import os
import sys
import numpy as np
import cv2
import shutil

from lib_inst_det.detector import Detector
from lib_inst_det.detectorAIR15 import DetectorAIR15
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

# _v2: 

# Demo code for detect and save object images from videos
# make same folder structure
if __name__ == '__main__':
    from_path = '../AIR_5yr/collected'
    to_path = '../AIR_5yr/cropped'

    # filename ex: object_1m_front_sit_back_light_kitchen_bottle_daeha_org_2020-07-16-16-46.mp4

    list_video_files = list_files(from_path, 'mp4')

    list_target_object = ['cup', 'remote']       # object in this list is saved as files

    num_images_remained = -1       # -1 means nothing to the generated images.
    do_structured_output = True

    # load the detector ans classifier
    detector = DetectorAIR15('../models', threshold=0.5)   # everything is under this path

    fid = open('../AIR_5yr/cropped/missing_file.txt', 'w')

    for object_name in list_target_object:
        list_instances = list_dirs(os.path.join(from_path, object_name))

        if object_name == 'cup':
            detector.display_classes = {
                'cup': '컵',
                # 'remote': '리모컨',
            }
        else:
            detector.display_classes = {
                # 'cup': '컵',
                'remote': '리모컨',
            }

        for instance_name in list_instances:
            # make saved directory

            if not os.path.exists(os.path.join(to_path, object_name, instance_name)):
                os.makedirs(os.path.join(to_path, object_name, instance_name))

            list_filename = list_files(os.path.join(from_path, object_name, instance_name), 'jpg')

            for filename in list_filename:
                image_np = cv2.imread(os.path.join(from_path, object_name, instance_name, filename))
                image_np = cv2.resize(image_np, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

                im2show = np.copy(image_np)

                detector.thresh = 0.9

                while True:
                    res_det = detector.detect(image_np)

                    if len(res_det) > 0:
                        break
                    elif detector.thresh < 0.4:
                        break
                    else:
                        detector.thresh -= 0.1

                if len(res_det) > 0:
                    im2show = OID.visualize(None, im2show, res_det)

                    # im2show_resized = cv2.resize(im2show, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                    # cv2.imshow('im2show', im2show)
                    # keyin = cv2.waitKey(0)

                    for object_index, item in enumerate(res_det):
                        bbox = item[0]
                        score = item[1]
                        class_name = item[2]

                        if class_name in list_target_object:
                            x1, y1, x2, y2 = bbox

                            # crop image
                            cropped_image = image_np[y1:y2, x1:x2, :]

                            # save in files
                            to_path_image = os.path.join(to_path, object_name, instance_name,
                                                         '%s_%01d_%.2f.png' % (filename.split('.')[0], object_index, score))

                            print(to_path_image)
                            cv2.imwrite(to_path_image, cropped_image)

                else:
                    fid.write('%s\n' % os.path.join(from_path, object_name, instance_name, filename))

        cv2.destroyAllWindows()

    fid.close()

        # if num_images_remained > 0:
        #     from_path_images = os.path.join(to_path, video_filename_only)
        #     list_image_files = list_files(from_path_images, 'png')
        #     list_image_files = sorted(list_image_files)
        #
        #     idx = np.round(np.linspace(0, len(list_image_files) - 1, num_images_remained)).astype(int)
        #     list_image_files = np.array(list_image_files)[idx.astype(int)]
        #
        #     # make new directory & shutil.copy
        #     to_path_images = os.path.join(to_path, video_filename_only + '_images_%d' % num_images_remained)
        #     if not os.path.exists(to_path_images):
        #         os.makedirs(to_path_images)
        #
        #     for item in list_image_files:
        #         shutil.copyfile(os.path.join(from_path_images, item), os.path.join(to_path_images, item))
        #
        #
        # if do_structured_output:
        #     # filename ex: object_1m_front_sit_back_light_kitchen_bottle_daeha_org_2020-07-16-16-46.mp4
        #     setting = video_filename_only.split('_')[:-4]
        #     object_name = video_filename_only.split('_')[-4]
        #     instance_name =video_filename_only.split('_')[-3]
        #
        #     if num_images_remained > 0:
        #         from_path_images = os.path.join(to_path, video_filename_only + '_images_%d' % num_images_remained)
        #     else:
        #         from_path_images = os.path.join(to_path, video_filename_only)
        #
        #     to_path_images = os.path.join(to_path + '_structured', '_'.join(setting), object_name, instance_name)
        #     if not os.path.exists(to_path_images):
        #         os.makedirs(to_path_images)
        #
        #     list_image_files = list_files(from_path_images, 'png')
        #     for item in list_image_files:
        #         shutil.copyfile(os.path.join(from_path_images, item), os.path.join(to_path_images, item))
