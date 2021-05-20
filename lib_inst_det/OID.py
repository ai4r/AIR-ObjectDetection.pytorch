import os
import pdb

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from lib_inst_det.detector import Detector
from lib_inst_det.detectorAIR15 import DetectorAIR15
from lib_inst_det.nnet import NNClassifier, GMMClassifier#, kNNClassifier
import datetime
import cv2
import numpy as np
from PIL import Image
import glob
from PIL import Image, ImageDraw, ImageFont
import random
import time
from lib_inst_det.gradCam import gradCam
from pytorch_grad_cam.utils.image import show_cam_on_image

# detector - faster-rcnn.pytorch
# classifier - prototype + cosine or linear fc
# Visualize - draw bbox
def resizeImagewithBorderRepeat(img_org, size, padding_type = cv2.BORDER_REPLICATE):
    if len(img_org.shape) == 2:
        c_o = 1
        (h_o, w_o) = img_org.shape
    else:
        (h_o, w_o, c_o) = img_org.shape

    size0 = max([h_o, w_o])

    border_l = int(round((size0 - w_o)/2))
    border_r = int(round(size0 - w_o - border_l))
    border_t = int(round(((size0 - h_o)/2)))
    border_b = int(round((size0 - h_o - border_t)))

    #   1. repeat to be square box
    if padding_type == cv2.BORDER_CONSTANT:
        if c_o == 1:
            img = cv2.copyMakeBorder(img_org, border_t, border_b, border_l, border_r, padding_type, (0))
        else:
            img = cv2.copyMakeBorder(img_org, border_t, border_b, border_l, border_r, padding_type, (0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img_org, border_t, border_b, border_l, border_r, padding_type)

    #   2. resize
    img = cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR)

    return img


def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


class OID(nn.Module):
    def __init__(self, baseFolder='models'):
        self.path_to_InstModel = os.path.join(baseFolder, 'InstModel')

        self.detector = Detector(baseFolder)           # load models and settings
        self.classifier = ModelManagerNN(self.path_to_InstModel)

    def detect(self, image):
        ret_bbox_score_class = self.detector.detect(image)

        # crop, classify, replace, result
        ret_bbox_score_inst = self.classifier.classify(image, ret_bbox_score_class)

        return ret_bbox_score_inst

    def register_prepare(self, category_name, instance_name):
        # delete and make folder
        print('delete all images of %s - %s' % (category_name, instance_name))
        self.classifier.delete_DB(category_name, instance_name)

    def register(self, image, category_name, instance_name):
        # detect the category_name object and save into image
        ret_bbox_score_class = self.detector.detect(image)

        # find the biggest bbox
        biggest_area = 0
        biggest_bbox = None
        ret = []
        for item in ret_bbox_score_class:
            [x1, y1, x2, y2] = item[0]
            # score = item[1]
            class_name = item[2]
            area = (x2 - x1) * (y2 - y1)

            if (class_name == category_name) and (area > biggest_area):
                biggest_bbox = [x1, y1, x2, y2]
                ret = [item]

        num_saved_images = 0
        if biggest_bbox is not None:
            # crop
            crop_image = image[biggest_bbox[1]:biggest_bbox[3], biggest_bbox[0]:biggest_bbox[2], :]
            self.classifier.save_DB(crop_image, category_name, instance_name)

        num_saved_images = len(list_files(os.path.join(self.path_to_InstModel, category_name, instance_name), 'png'))

        return ret, num_saved_images

    def register_finish(self):
        self.classifier.reload_DB()


    def visualize(self, image, list_info, thresh=0.8, rect_color=(0, 204, 0), text_color=(255, 255, 255)):
        """Visual debugging of detections."""

        print(list_info)

        for item in list_info:
            # item has 4 elements: bbox(4)_score(1)_class(str)
            # item has 6 elements: bbox(4)_score(1)_class(str)_score(1)_instance(str)
            bbox = item[0]
            score = item[1]
            class_name = item[2]
            if score > thresh:
                cv2.rectangle(image, bbox[0:2], bbox[2:4], rect_color, 2)

                if len(item) < 4:
                    strResult = '%s: %.3f' % (class_name, score)
                else:
                    score_i = item[3]
                    inst_name = item[4]

                    strResult = '%s: %.3f (%s: %.3f)' % (class_name, score, inst_name, score_i)

                cv2.putText(image, strResult, (bbox[0], bbox[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0, text_color, thickness=2)

        return image

class OIDv2(nn.Module):
    def __init__(self, baseFolder='models'):
        self.path_to_InstModel = os.path.join(baseFolder, 'InstModel')

        self.use_AIR15_detector = True
        self.use_COCO_detector = False

        self.map_classname_to_korean = {}

        # if self.use_comment_generator:
        #     self.use_COCO_detector = True
        #     print('self.use_COCO_detector is turned on for comment_generator!')

        if self.use_COCO_detector:
            self.detectorCOCO5 = Detector(baseFolder)           # load models and settings
            self.map_classname_to_korean.update(**self.detectorCOCO5.display_classes)
        else:
            self.detectorCOCO5 = None

        if self.use_AIR15_detector:
            self.detectorAIR15 = DetectorAIR15(baseFolder)  # load models and settings
            self.map_classname_to_korean.update(**self.detectorAIR15.display_classes)
        else:
            self.detectorAIR15 = None

        if not self.use_COCO_detector and not self.use_AIR15_detector:
            print('Both detectors are not activated. Please check it.')

        # classifier
        self.classifier = ModelManagerNN(self.path_to_InstModel, detection_model=self.detectorAIR15)    # det_model is used to calcuate att_map

        # 0. predefined variables
        self.list_obj_and_comment = {}
        self.list_obj_and_comment.update({'mobile_phone': dict.fromkeys([4, 5], '핸드폰을 들고 다니시면 떨어뜨릴 수 있어요. 주머니에 넣고 다니세요.')})
        self.list_obj_and_comment.update({'key': dict.fromkeys([4, 5], '열쇠를 들고 다니시면 잃어버릴 수 있어요. 주머니에 넣고 다니세요.')})
        self.list_obj_and_comment.update({'wallet': dict.fromkeys([4, 5], '지갑을 들고 다니시면 잃어버릴 수 있어요. 주머니에 넣고 다니세요.')})
        self.list_obj_and_comment.update({'pack': dict.fromkeys([1, 2, 3, 4, 5], '담배보다 껌이나 사탕은 어떠세요?')})
        self.list_obj_and_comment.update({'medicine_case': dict.fromkeys([1, 2, 3], '약이 맞는지 확인하고 드세요.')})
        self.list_obj_and_comment.update({'medicine_packet': dict.fromkeys([1, 2, 3], '약이 맞는지 확인하고 드세요.')})
        self.list_obj_and_comment.update({'hat': dict.fromkeys([4, 5], '모자가 잘 어울리세요.')})

    #         self.list_obj_and_comment.update({'tie': {4: '중요한 자리에 가시나봐요. 안녕히 다녀오세요.',
    #                                              5: '중요한 자리, 잘 다녀오셨나요? 오늘 하루도 고생하셨습니다.'}
    #                                     })
    #         self.list_obj_and_comment.update({'umbrella': {4: '우산을 가져가시네요. 비오는 중에는 앞을 꼭 보며 걸으셔야해요.',
    #                                                   5: '비오는 중에, 안녕히 다녀오셨나요?'}
    #                                     })

    def detect(self, image):
        if self.use_COCO_detector:
            ret_bbox_score_class_COCO5 = self.detectorCOCO5.detect(image)
        else:
            ret_bbox_score_class_COCO5 = []

        if self.use_AIR15_detector:
            ret_bbox_score_class_AIR15 = self.detectorAIR15.detect(image)
        else:
            ret_bbox_score_class_AIR15 = []

        if not self.use_COCO_detector and not self.use_AIR15_detector:
            print('Both detectors are not activated. Please check it.')

        ret_bbox_score_class = ret_bbox_score_class_COCO5 + ret_bbox_score_class_AIR15
        # crop, classify, replace, result

        ret_bbox_score_inst = self.classifier.classify(image, ret_bbox_score_class)
        # ret_bbox_score_inst = ret_bbox_score_class_COCO5 + ret_bbox_score_inst_AIR15

        return ret_bbox_score_inst

    def register_prepare(self, category_name, instance_name):
        # delete and make folder
        print('delete all images of %s - %s' % (category_name, instance_name))
        self.classifier.delete_DB(category_name, instance_name)

    def register(self, image, category_name, instance_name):
        import pdb
        # detect the category_name object and save into image
        ret_bbox_score_class_AIR15 = self.detectorAIR15.detect(image)
        ret_bbox_score_class_COCO = self.detectorCOCO5.detect(image)
        ret_bbox_score_class = ret_bbox_score_class_AIR15 + ret_bbox_score_class_COCO

        # find the biggest bbox
        biggest_area = 0
        biggest_bbox = None
        ret = []
        for item in ret_bbox_score_class:
            [x1, y1, x2, y2] = item[0]
            # score = item[1]
            class_name = item[2]
            area = (x2 - x1) * (y2 - y1)

            if (class_name == category_name) and (area > biggest_area):
                biggest_bbox = [x1, y1, x2, y2]
                ret = [item]

        num_saved_images = 0
        if biggest_bbox is not None:
            # crop
            crop_image = image[biggest_bbox[1]:biggest_bbox[3], biggest_bbox[0]:biggest_bbox[2], :]
            self.classifier.save_DB(crop_image, category_name, instance_name)

        num_saved_images = len(list_files(os.path.join(self.path_to_InstModel, category_name, instance_name), 'png'))

        return ret, num_saved_images

    def register_finish(self):
        self.classifier.reload_DB()


    def visualize(self, image, list_info, box_color=(0, 204, 0), text_color=(255, 255, 255),
                                   text_bg_color=(0, 204, 0), fontsize=20, thresh=0.8, draw_score=True,
                                   draw_text_out_of_box=True, map_classname_to_korean=None):
        """Visual debugging of detections."""
        print(list_info)

        font = ImageFont.truetype('NanumGothic.ttf', fontsize)
        image_pil = Image.fromarray(image)
        image_draw = ImageDraw.Draw(image_pil)

        for item in list_info:
            # item has 4 elements: bbox(4)_score(1)_class(str)
            # item has 6 elements: bbox(4)_score(1)_class(str)_score(1)_instance(str)
            bbox = item[0]
            score = item[1]
            class_name = item[2]

            if map_classname_to_korean is not None:
                # red: big object
                # green: handheld object w/o owner
                # blue: handheld object w owner
                if class_name in ['tv', 'refrigerator', 'couch', 'bed']:
                    box_color = text_bg_color = (0, 0, 204)
                else:
                    box_color = text_bg_color = (0, 204, 0)

                if class_name in map_classname_to_korean.keys():
                    class_name = map_classname_to_korean[class_name]

            if score > thresh:
                image_draw.rectangle(bbox, outline=box_color, width=3)

                if len(item) < 4:
                    if draw_score:
                        strText = '%s: %.3f' % (class_name, score)
                    else:
                        strText = class_name
                else:
                    box_color = text_bg_color = (204, 0, 0)
                    score_i = item[3]
                    inst_name = item[4]

                    if draw_score:
                        strText = '%s: %.3f (%s: %.3f)' % (class_name, score, inst_name, score_i)
                    else:
                        strText = '%s (%s)' % (class_name, inst_name)

                text_w, text_h = font.getsize(strText)

                if draw_text_out_of_box:
                    image_draw.rectangle((bbox[0], bbox[1] - 20, bbox[0] + text_w, bbox[1] - 20 + text_h),
                                      fill=text_bg_color)
                    image_draw.text((bbox[0], bbox[1] - 20), strText, font=font, fill=text_color)
                else:
                    image_draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=text_bg_color)
                    image_draw.text((bbox[0], bbox[1]), strText, font=font, fill=text_color)

        image = np.array(image_pil)

        return image

    def generate_comment(self, person_bbox, list_objs_bbox_score_class, context):
        # person_bbox: [x1, y1, x2, y2]
        # list list_objs_bbox_score_class: each item has [x1, y1, x2, y2, score, classname]
        # context: 0 : undefined,1 : in the morning, 2 : in the noon,3 : in the night, 4 : when go out (외출할 때), 5 : when come in (외출후 들어올 때), 9: first seeing (처음 본 사람)

        def calculate_iou(bboxA, bboxB):
            inter_x1 = np.maximum(bboxA[:2], bboxB[:2])
            inter_x2 = np.minimum(bboxA[2:], bboxB[2:])

            inter_area = max(inter_x2[0] - inter_x1[0], 0) * max(inter_x2[1] - inter_x1[1], 0)
            bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
            bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])

            iou = float(inter_area) / float(bboxA_area + bboxB_area - inter_area)

            return iou

        # 1. first check the position and name of object bboxes
        # import pdb
        # pdb.set_trace()

        list_candidate_comments = []
        for iobj, obj_bbox_score_class in enumerate(list_objs_bbox_score_class):
            # item has 4 elements: bbox(4)_score(1)_class(str)
            # item has 6 elements: bbox(4)_score(1)_class(str)_score(1)_instance(str)
            obj_bbox = obj_bbox_score_class[0]
            # obj_score = obj_bbox_score_class[1]
            obj_class = obj_bbox_score_class[2]

            obj_cx = (obj_bbox[0] + obj_bbox[2]) / 2
            obj_cy = (obj_bbox[1] + obj_bbox[3]) / 2

            # check object's center is in person_bbox
            if obj_cx > person_bbox[0] and obj_cx < person_bbox[2] and obj_cy > person_bbox[1] and obj_cy < person_bbox[3]:

                # check object is in comment_object_list:
                try:
                    list_candidate_comments.append(self.list_obj_and_comment[obj_class][context])
                except:
                    pass

        # 2. randomly select one comment
        ret_comment = None
        if len(list_candidate_comments) > 0:
            ret_comment = random.choice(list_candidate_comments)

        return ret_comment


# Manager between DB <-> object classifier
class ModelManagerNN():
    def __init__(self, instBase_path, num_hid_ftr=2048, fix_base_net=True, num_layer=1, fc_type='linear',
                 force_to_train=False, detection_model=None):
        self.modelList = []
        # self.modelListGMM = []
        # self.modelListkNN = []
        self.modelCategorynameList = []
        self.modelInstancenameList = []
        self.instBase_path = instBase_path

        # debug image
        self.save_debug_image_cropped = True
        self.save_debug_image_whole = False
        self.save_debug_image_path = '/home/yochin/Desktop/AIR-ObjectDetection.pytorch/debug_image_folder'

        # NN options
        self.num_hid_ftr = num_hid_ftr
        self.fix_base_net = fix_base_net
        self.num_layer = num_layer
        self.fc_type = fc_type
        self.lr = 0.01

        # do load_DB
        self.force_to_train = force_to_train
        self.load_DB(self.instBase_path)

        if detection_model is not None:
            print('gradCam is loaded!!!!')
            self.detection_model_categories = detection_model.classes
            self.gradCam = gradCam(detection_model)
        else:
            self.gradCam = None

        self.do_not_checksum = True

    tr_transforms = transforms.Compose([
                            transforms.Scale(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(0.5),
                            # transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))
                        ])
    ts_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),

        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # tr_transforms_GMM = transforms.Compose([
    #     transforms.Scale(24),
    #     transforms.RandomCrop(16),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5),
    #                          (0.5, 0.5, 0.5))
    # ])
    # ts_transforms_GMM = transforms.Compose([
    #     transforms.Scale(24),
    #     transforms.RandomCrop(16),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5),
    #                          (0.5, 0.5, 0.5))
    # ])


    def reload_DB(self):
        self.modelList = []
        # self.modelListGMM = []
        # self.modelListkNN = []
        self.modelCategorynameList = []
        self.modelInstancenameList = []

        self.load_DB(self.instBase_path)


    def load_DB(self, path, category_index=None, verbose=True):
        # 1. try to load a model
        # 2. if they are same, then no further train
        # 3. if they are not same, then re-train again
        if not os.path.exists(path):
            print('object instance path {} does not exists. generate it!'.format(path))
            os.makedirs(path)

        listCategory = os.listdir(path)
        for iC, objCate in enumerate(listCategory):        # object category
            listInst = [item for item in os.listdir(os.path.join(path, objCate)) if os.path.isdir(os.path.join(path, objCate, item))]
            nb_classes = len(listInst)

            nnC = NNClassifier(objCate,
                               num_hid_ftr=self.num_hid_ftr,
                               num_class=nb_classes,
                               fix_base_net=self.fix_base_net,
                               num_layer=self.num_layer,
                               fc_type=self.fc_type)  # construct classifier

            # gmmC = GMMClassifier(objCate,
            #                      num_cluster=1,
            #                      num_class=nb_classes)
            #
            #
            # # knnC = kNNClassifier(objCate,
            # #                      num_class=nb_classes)

            # try to load a model
            pth_path = os.path.join(path, objCate, 'model_frz%d_layer%d_nhid%d_fc%s_class%d.pt' % (self.fix_base_net, self.num_layer, self.num_hid_ftr, self.fc_type, nb_classes))

            if self.force_to_train:
                # suc = [False, False, False]
                suc = [False]
            else:
                suc1 = nnC.load(pth_path, nb_classes)        # fail when no file or numInstance in not same
                # suc2 = gmmC.load(os.path.join(path, objCate, 'gmmC.pkl'))
                # suc3 = knnC.load(os.path.join(path, objCate, 'knnC.pkl'))
                # suc = [suc1, suc2, suc3]
                # suc = [suc1, suc2]
                suc = [suc1]

            # TODO: check transform is right?
            dataset = dset.ImageFolder(root=os.path.join(path, objCate), transform=self.tr_transforms)
            # dataset_GMM = dset.ImageFolder(root=os.path.join(path, objCate), transform=self.tr_transforms_GMM)

            if all(suc) == False:
                print('Fail to read the model file. Train the network')

                if verbose:
                    print('[%s] has %d instances: '%(objCate, nb_classes), listInst)

                # dataX: n_data x dim (102400)
                # dataY: list
                if not suc[0]:
                    nnC.train(dataset, learning_rate=self.lr, stop_acc=0.90)             # train classifier
                    nnC.save(pth_path) # save the model
                # if not suc[1]:
                #     gmmC.train(dataset_GMM)  # train classifier
                #     gmmC.save(pth_path)  # save the model
                # if not suc[2]:
                #     knnC.train(dataset)  # train classifier
                #     knnC.save(pth_path)  # save the model
            else:
                print('Success to read the model file')
                print('\tpath: ', pth_path)
                print('\tcategory: ', objCate)
                print('\tinstances (%d): ' % nb_classes, listInst)

            self.modelList.append(nnC)      # add NNs to the list
            # self.modelListGMM.append(gmmC)  # add NNs to the list
            # self.modelListkNN.append(knnC)  # add NNs to the list
            self.modelCategorynameList.append(objCate)
            self.modelInstancenameList.append(dataset.class_to_idx)
            print('\tclass_to_index:', dataset.class_to_idx)

            if self.save_debug_image_cropped or self.save_debug_image_whole:
                for item in listInst:
                    if not os.path.exists(os.path.join(self.save_debug_image_path, item)):
                        os.makedirs(os.path.join(self.save_debug_image_path, item))


    def classify(self, image, bbox_score_class):
        ret_bbox_score_inst = []

        for item in bbox_score_class:
            # item: [0] bbox, [1] score, [2] category_name
            [x1, y1, x2, y2] = item[0]
            c_name = item[2]

            nameSimilarInst = ''
            probSimilarInst = 0.

            if False:
                # 1. crop bbox_score_class
                crop_w_half = int((x2 - x1) / 4)  # add 25% to left and 25% to right
                crop_h_half = int((y2 - y1) / 4)

                crop_x1 = max(x1 - crop_w_half, 0)
                crop_x2 = min(x2 + crop_w_half, image.shape[1])

                crop_y1 = max(y1 - crop_h_half, 0)
                crop_y2 = min(y2 + crop_h_half, image.shape[1])

                image_crop_for_attention = image[crop_y1:crop_y2, crop_x1:crop_x2, :]
                image_crop_for_clf = image[y1:y2, x1:x2, :]

                PATCH_SIZE = 224
                scale_resized = 224.0 / float(min(crop_x2-crop_x1, crop_y2-crop_y1))

                PATCH_MARGIN_SIZE = int(PATCH_SIZE / 6)

                if self.gradCam is not None:
                    image_crop_bgr_224 = cv2.resize(image_crop_for_attention, dsize=None, fx=scale_resized, fy=scale_resized)
                    cv2.imshow('./%s_input.png' % c_name, image_crop_bgr_224)

                    image_crop_bgr_224_tensor = self.gradCam.convert_to_tensor(image_crop_bgr_224)  # bgr -> tensor
                    score = self.gradCam.get_score(image_crop_bgr_224_tensor)   # tensor -> class
                    selected_class_by_cam = self.detection_model_categories[score.argmax()]
                    # print(score)
                    print('--> selected class is ', selected_class_by_cam, score.max())

                    if selected_class_by_cam == c_name:
                        id_idx = np.where(self.detection_model_categories == c_name)
                        gradMap = self.gradCam.get_gradCam(image_crop_bgr_224_tensor, target_id=id_idx[0])  # get gradMap
                        gradMap = gradMap[0, :]
                        visualization = show_cam_on_image(image_crop_bgr_224 / 255., gradMap)
                        cv2.imshow('./%s_gradcam_%s.png' % (c_name, c_name), visualization)

                        # cut margin and resize
                        cv2.imshow('./%s_intput_object_%s.png' % (c_name, c_name), image_crop_for_clf)

                        # gradMap = gradMap[PATCH_MARGIN_SIZE:-PATCH_MARGIN_SIZE, PATCH_MARGIN_SIZE:-PATCH_MARGIN_SIZE]
                        # gradMap_resized = cv2.resize(gradMap, (x2-x1, y2-y1))
                        # image_crop_for_clf = image_crop_for_clf * np.expand_dims(gradMap_resized, axis=2)
                        # image_crop_for_clf = image_crop_for_clf.astype(np.uint8)
                        # cv2.imshow('./%s_gradcamed_object_%s.png' % (c_name, c_name), image_crop_for_clf)
                        #
                        # del gradMap, visualization
                        # del image_crop_bgr_224, image_crop_bgr_224_tensor, score

                        cv2.waitKey(0)

                        # image_crop_resized = cv2.cvtColor(image_crop_for_clf, cv2.COLOR_BGR2RGB)
                        # image_crop_resized_pil = Image.fromarray(image_crop_resized)
                        # image_crop_resized_tensor = self.ts_transforms(image_crop_resized_pil)
                        # image_crop_resized_tensor = image_crop_resized_tensor.unsqueeze_(0).cuda()
                        # # image_crop_resized_tensor = image_crop_resized_tensor.cuda()
                        #
                        # del image_crop_resized_tensor

            if c_name in self.modelCategorynameList:   # classifier
                # 1. crop bbox_score_class
                crop_w_half = int((x2 - x1) / 4)  # add 25% to left and 25% to right
                crop_h_half = int((y2 - y1) / 4)

                crop_x1 = max(x1 - crop_w_half, 0)
                crop_x2 = min(x2 + crop_w_half, image.shape[1])

                crop_y1 = max(y1 - crop_h_half, 0)
                crop_y2 = min(y2 + crop_h_half, image.shape[1])

                image_crop_for_attention = image[crop_y1:crop_y2, crop_x1:crop_x2, :]
                image_crop_for_clf = image[y1:y2, x1:x2, :]

                scale_resized = 224.0 / float(min(crop_x2 - crop_x1, crop_y2 - crop_y1))

                # PATCH_SIZE = 224

                if self.gradCam is not None:
                    # image_crop_bgr_224 = cv2.resize(image_crop_for_attention, dsize=(PATCH_SIZE, PATCH_SIZE))
                    image_crop_bgr_224 = cv2.resize(image_crop_for_attention, dsize=None, fx=scale_resized, fy=scale_resized)
                    cv2.imshow('./%s_input.png' % c_name, image_crop_bgr_224)

                    image_crop_bgr_224_tensor = self.gradCam.convert_to_tensor(image_crop_bgr_224)  # bgr -> tensor
                    score = self.gradCam.get_score(image_crop_bgr_224_tensor)   # tensor -> class
                    selected_class_by_cam = self.detection_model_categories[score.argmax()]
                    # print(score)
                    print('--> selected class is ', selected_class_by_cam, score.max())

                    if selected_class_by_cam == c_name or self.do_not_checksum:
                        id_idx = np.where(self.detection_model_categories == c_name)
                        gradMap = self.gradCam.get_gradCam(image_crop_bgr_224_tensor, target_id=id_idx[0])  # get gradMap
                        gradMap = gradMap[0, :]
                        visualization = show_cam_on_image(image_crop_bgr_224 / 255., gradMap)
                        cv2.imshow('./%s_gradcam_%s.png' % (c_name, c_name), visualization)

                        # cut margin and resize
                        cv2.imshow('./%s_intput_object_%s.png' % (c_name, c_name), image_crop_for_clf)

                        # pdb.set_trace()
                        PATCH_MARGIN_SIZE_H = int(gradMap.shape[0] / 6)
                        PATCH_MARGIN_SIZE_W = int(gradMap.shape[1] / 6)

                        gradMap = gradMap[PATCH_MARGIN_SIZE_H:-PATCH_MARGIN_SIZE_H, PATCH_MARGIN_SIZE_W:-PATCH_MARGIN_SIZE_W]
                        gradMap_resized = cv2.resize(gradMap, (x2-x1, y2-y1))
                        image_crop_for_clf = image_crop_for_clf * np.expand_dims(gradMap_resized, axis=2)
                        image_crop_for_clf = image_crop_for_clf.astype(np.uint8)
                        cv2.imshow('./%s_gradcamed_object_%s.png' % (c_name, c_name), image_crop_for_clf)

                        del gradMap, visualization
                        del image_crop_bgr_224, image_crop_bgr_224_tensor, score

                        cv2.waitKey(0)

                        image_crop_resized = cv2.cvtColor(image_crop_for_clf, cv2.COLOR_BGR2RGB)
                        image_crop_resized_pil = Image.fromarray(image_crop_resized)
                        image_crop_resized_tensor = self.ts_transforms(image_crop_resized_pil)
                        image_crop_resized_tensor = image_crop_resized_tensor.unsqueeze_(0).cuda()
                        # image_crop_resized_tensor = image_crop_resized_tensor.cuda()

                        # 2. pass it to the classifier
                        i_clf = self.modelCategorynameList.index(c_name)

                        inst_idx_NN, inst_prob_NN = self.modelList[i_clf].inference(image_crop_resized_tensor)
                        # inst_idx_GMM, inst_prob_GMM = self.modelListGMM[i_clf].inference(image_crop_resized_tensor_GMM)

                        # yochin debug
                        inst_idx = inst_idx_NN
                        inst_prob = inst_prob_NN

                        # 3. get the result and append it to the list
                        # nameSimilarInst = self.modelInstancenameList[i_clf][inst_idx]
                        nameSimilarInst = list(self.modelInstancenameList[i_clf].keys())[list(self.modelInstancenameList[i_clf].values()).index(inst_idx)]
                        probSimilarInst = inst_prob

                        # print('NN result: %s %f' % (nameSimilarInst, probSimilarInst))

                        if self.save_debug_image_cropped:
                            ctime = time.time() * 1000
                            cv2.imwrite(os.path.join(self.save_debug_image_path, nameSimilarInst, '{}_{:.4f}_{}.png'.format(nameSimilarInst, probSimilarInst.item(), ctime)), cv2.cvtColor(image_crop_resized, cv2.COLOR_RGB2BGR))

                        del image_crop_resized_tensor
                    else:
                        nameSimilarInst = 'unknown'
                        probSimilarInst = 0.0

            if nameSimilarInst != '':
                item.extend([probSimilarInst, nameSimilarInst])

            ret_bbox_score_inst.append(item)

        if self.save_debug_image_whole:
            ctime = time.time() * 1000
            cv2.imwrite(os.path.join(self.save_debug_image_path, 'image_{}.png'.format(ctime)), image)  # BGR


        return ret_bbox_score_inst


    def delete_DB(self, category_name, instance_name):
        path_to_instance = os.path.join(self.instBase_path, category_name, instance_name)

        # make blank directory
        if not os.path.exists(path_to_instance):
            os.makedirs(path_to_instance)

        # delete files
        files = glob.glob(os.path.join(path_to_instance, '*'))
        for f in files:
            os.remove(f)

        # delete model
        files = glob.glob(os.path.join(self.instBase_path, category_name, '*.pt'))
        for f in files:
            os.remove(f)


    def save_DB(self, image_crop, category_name, instance_name):
        pathDB = os.path.join(self.instBase_path, category_name, instance_name)
        if not os.path.exists(pathDB):
            os.makedirs(pathDB)

        strTimestamp = datetime.datetime.now().isoformat()
        pathSaveTimestamp = os.path.join(pathDB, '%s' % strTimestamp)

        # cv2.imshow('cropped', image_np[top:bottom, left:right, :])
        cv2.imwrite(pathSaveTimestamp + '.png', image_crop)

