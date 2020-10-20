import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from lib_inst_det.detector import Detector
from lib_inst_det.nnet import NNClassifier
import datetime
import cv2
import numpy as np
from PIL import Image
import glob

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
            crop_image = image[y1:y2, x1:x2, :]
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


# Manager between DB <-> object classifier
class ModelManagerNN():
    def __init__(self, instBase_path, num_hid_ftr=2048, fix_base_net=True, num_layer=1, fc_type='linear', force_to_train=False):
        self.modelList = []
        self.modelCategorynameList = []
        self.modelInstancenameList = []
        self.instBase_path = instBase_path

        # NN options
        self.num_hid_ftr = num_hid_ftr
        self.fix_base_net = fix_base_net
        self.num_layer = num_layer
        self.fc_type = fc_type
        self.lr = 0.01

        # do load_DB
        self.force_to_train = force_to_train
        self.load_DB(self.instBase_path)

    tr_transforms = transforms.Compose([
                            transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))
                        ])
    ts_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])


    def reload_DB(self):
        self.modelList = []
        self.modelCategorynameList = []
        self.modelInstancenameList = []

        self.load_DB(self.instBase_path)


    def load_DB(self, path, category_index=None, verbose=True):
        # 1. try to load a model
        # 2. if they are same, then no further train
        # 3. if they are not same, then re-train again
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

            # try to load a model
            pth_path = os.path.join(path, objCate, 'model_frz%d_layer%d_nhid%d_fc%s_class%d.pt' % (self.fix_base_net, self.num_layer, self.num_hid_ftr, self.fc_type, nb_classes))

            if self.force_to_train:
                suc = False
            else:
                suc = nnC.load(pth_path, nb_classes)        # fail when no file or numInstance in not same

            # TODO: check transform is right?
            dataset = dset.ImageFolder(root=os.path.join(path, objCate), transform=self.tr_transforms)

            if suc == False:
                print('Fail to read the model file. Train the network')

                if verbose:
                    print('[%s] has %d instances: '%(objCate, nb_classes), listInst)

                # dataX: n_data x dim (102400)
                # dataY: list
                nnC.train(dataset, learning_rate=self.lr)             # train classifier

                # save the model
                nnC.save(pth_path)
            else:
                print('Success to read the model file')
                print('\tpath: ', pth_path)
                print('\tcategory: ', objCate)
                print('\tinstances (%d): ' % nb_classes, listInst)

            self.modelList.append(nnC)      # add NNs to the list
            self.modelCategorynameList.append(objCate)
            self.modelInstancenameList.append(dataset.class_to_idx)
            print('\tclass_to_index:', dataset.class_to_idx)


    def classify(self, image, bbox_score_class):
        ret_bbox_score_inst = []

        for item in bbox_score_class:
            # item: [0] bbox, [1] score, [2] category_name
            [x1, y1, x2, y2] = item[0]
            c_name = item[2]

            nameSimilarInst = ''
            probSimilarInst = 0.

            if c_name in self.modelCategorynameList:
                # 1. crop bbox_score_class
                image_crop = image[y1:y2, x1:x2, :]
                # image_crop_resized = cv2.resize(image_crop, (64, 64), None, interpolation=cv2.INTER_LINEAR)
                # image_crop_resized = resizeImagewithBorderRepeat(image_crop, 64)
                # cv2.imwrite('./ttt.png', image_crop_resized)
                image_crop_resized = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                image_crop_resized_pil = Image.fromarray(image_crop_resized)
                image_crop_resized_tensor = self.ts_transforms(image_crop_resized_pil)
                image_crop_resized_tensor.unsqueeze_(0)
                image_crop_resized_tensor = image_crop_resized_tensor.cuda()

                # 2. pass it to the classifier
                i_clf = self.modelCategorynameList.index(c_name)

                inst_idx, inst_prob = self.modelList[i_clf].inference(image_crop_resized_tensor)

                # 3. get the result and append it to the list
                # nameSimilarInst = self.modelInstancenameList[i_clf][inst_idx]
                nameSimilarInst = list(self.modelInstancenameList[i_clf].keys())[list(self.modelInstancenameList[i_clf].values()).index(inst_idx)]
                probSimilarInst = inst_prob

                # print('NN result: %s %f' % (nameSimilarInst, probSimilarInst))

            item.extend([probSimilarInst, nameSimilarInst])
            ret_bbox_score_inst.append(item)

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

