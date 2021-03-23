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


# Demo code for detect and save object images from videos
# make same folder structure
if __name__ == '__main__':
    # to_path = 'AIR_4yr/test/two_objects'
    # from_path = 'AIR_4yr/test'

    # to_path = 'AIR_4yr_2nd/object_recog_(reg_test_video)_20200716/register_videos/images'
    # from_path = 'AIR_4yr_2nd/object_recog_(reg_test_video)_20200716/register_videos'

    to_path = 'AIR_4yr_2nd/recog_videos/images'
    from_path = 'AIR_4yr_2nd/recog_videos/videos'
    # filename ex: object_1m_front_sit_back_light_kitchen_bottle_daeha_org_2020-07-16-16-46.mp4

    list_video_files = list_files(from_path, 'mp4')

    list_target_object = ['bottle', 'cell phone', 'cup', 'remote']       # object in this list is saved as files

    # frame rate
    # 30fps
    # 1 img/s -> 30
    # 2 img/s -> 15
    # 5 img/s -> 6
    frame_rate = 6
    num_images_remained = -1       # -1 means nothing to the generated images.
    do_structured_output = True

    # load the detector ans classifier
    detector = Detector('models', threshold=0.5)   # everything is under this path

    for video_filename in list_video_files:
        # make saved directory
        # _, video_filename = os.path.split(from_path_video)
        from_path_video = os.path.join(from_path, video_filename)
        video_filename_only = os.path.splitext(video_filename)[0]

        if not os.path.exists(os.path.join(to_path, video_filename_only)):
            os.makedirs(os.path.join(to_path, video_filename_only))

        cap = cv2.VideoCapture(from_path_video)

        if not cap.isOpened():
            raise RuntimeError("Video file could not be opened. Please check the file.", from_path_video)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

                if frame_index % frame_rate == 0:
                    image_np = np.array(frame)
                    im2show = np.copy(image_np)

                    res_det = detector.detect(image_np)

                    for object_index, item in enumerate(res_det):
                        bbox = item[0]
                        score = item[1]
                        class_name = item[2]

                        if class_name in list_target_object:
                            x1, y1, x2, y2 = bbox

                            # crop image
                            cropped_image = image_np[y1:y2, x1:x2, :]

                            # save in files
                            to_path_image = os.path.join(to_path, video_filename_only, '%05d_%02d.png' % (frame_index, object_index))

                            # print(to_path_image)
                            cv2.imwrite(to_path_image, cropped_image)

                    im2show = OID.visualize(None, im2show, res_det)

                    cv2.imshow('im2show', im2show)
                    cv2.waitKey(10)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


        if num_images_remained > 0:
            from_path_images = os.path.join(to_path, video_filename_only)
            list_image_files = list_files(from_path_images, 'png')
            list_image_files = sorted(list_image_files)

            idx = np.round(np.linspace(0, len(list_image_files) - 1, num_images_remained)).astype(int)
            list_image_files = np.array(list_image_files)[idx.astype(int)]

            # make new directory & shutil.copy
            to_path_images = os.path.join(to_path, video_filename_only + '_images_%d' % num_images_remained)
            if not os.path.exists(to_path_images):
                os.makedirs(to_path_images)

            for item in list_image_files:
                shutil.copyfile(os.path.join(from_path_images, item), os.path.join(to_path_images, item))


        if do_structured_output:
            # filename ex: object_1m_front_sit_back_light_kitchen_bottle_daeha_org_2020-07-16-16-46.mp4
            setting = video_filename_only.split('_')[:-4]
            object_name = video_filename_only.split('_')[-4]
            instance_name =video_filename_only.split('_')[-3]

            if num_images_remained > 0:
                from_path_images = os.path.join(to_path, video_filename_only + '_images_%d' % num_images_remained)
            else:
                from_path_images = os.path.join(to_path, video_filename_only)

            to_path_images = os.path.join(to_path + '_structured', '_'.join(setting), object_name, instance_name)
            if not os.path.exists(to_path_images):
                os.makedirs(to_path_images)

            list_image_files = list_files(from_path_images, 'png')
            for item in list_image_files:
                shutil.copyfile(os.path.join(from_path_images, item), os.path.join(to_path_images, item))
