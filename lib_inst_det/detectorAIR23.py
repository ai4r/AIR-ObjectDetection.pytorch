# This code is copied and modified from https://github.com/jwyang/faster-rcnn.pytorch/blob/master/demo.py

import _init_paths
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cv2

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread   # scipy.__version__ < 1.2.0
from imageio import imread  # new

from model.utils.config import cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
import pdb
import numpy as np
import argparse
import pprint
from PIL import Image, ImageDraw, ImageFont

# every complex things are in this class
# user needs just to call the APIs.
# detector - faster-rcnn.pytorch
class DetectorAIR23():
    def __init__(self, baseFolder='models', filename='faster_rcnn_1_10_9999_mosaicCL3to5_CBAM_Gblur_class23_wOrgCW.pth',
                 threshold=0.9, att_type='CBAM'): # att_type=None
        super(DetectorAIR23, self).__init__()

        self.cfg = __import__('model').utils.config.cfg

        def parse_args():
            """
            Parse input arguments
            """
            parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfgs/vgg16.yml', type=str)
            parser.add_argument('--net', dest='net',
                                help='vgg16, res50, res101, res152',
                                default='res101', type=str)
            parser.add_argument('--set', dest='set_cfgs',
                                help='set config keys', default=None,
                                nargs=argparse.REMAINDER)
            parser.add_argument('--cuda', dest='cuda',
                                help='whether use CUDA',
                                action='store_true')
            parser.add_argument('--mGPUs', dest='mGPUs',
                                help='whether use multiple GPUs',
                                action='store_true')
            parser.add_argument('--cag', dest='class_agnostic',
                                help='whether perform class_agnostic bbox regression',
                                action='store_true')
            parser.add_argument('--parallel_type', dest='parallel_type',
                                help='which part of model to parallel, 0: all, 1: model before roi pooling',
                                default=0, type=int)
            parser.add_argument('--ls', dest='large_scale',
                                help='whether use large imag scale',
                                action='store_true')
            parser.add_argument('--use_FPN', dest='use_FPN', action='store_true')

            return parser

        cmd_args = [
            '--net', 'res101',
            '--ls',
            '--cuda',
            '--use_FPN',
        ]

        load_name = os.path.join(baseFolder, filename) # w/o bottle class

        self.thresh = threshold

        parser = parse_args()
        self.args = parser.parse_args(cmd_args)

        self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        self.args.cfg_file = "{}/cfgs/{}_ls.yml".format(baseFolder, self.args.net) if self.args.large_scale else "{}/cfgs/{}.yml".format(baseFolder, self.args.net)

        print('Called with args:')
        print(self.args)

        if self.args.cfg_file is not None:
            # check cfg file and copy
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)

        self.cfg.USE_GPU_NMS = self.args.cuda

        print('Using config:')
        pprint.pprint(self.cfg)
        np.random.seed(self.cfg.RNG_SEED)

        # train set
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        #
        # input_dir = self.args.load_dir + "/" + self.args.net + "/" + self.args.dataset
        # if not os.path.exists(input_dir):
        #     raise Exception('There is no input directory for loading network from ' + input_dir)
        # load_name = os.path.join(input_dir,
        #                          'faster_rcnn_{}_{}_{}.pth'.format(self.args.checksession, self.args.checkepoch, self.args.checkpoint))

        self.classes = np.asarray([
            '__background__',  # always index 0
             'cane_stick',
             'mobile_phone',
             'pack',
             'cup',
             'glasses',
             'hat',
             'key',
             'medicine_case',
             'medicine_packet',
             'newspaper',
             'remote',
             'sock',
             'towel',
             'wallet',
             'pen',
             'sink',
             'table',
             'bed',
             'sofa_bed',
             'refrigerator',
             'television',
             'toilet',
             'mechanical_fan',
        ])

        # self.display_classes = self.classes
        self.display_classes = {
            'cup': '컵',
            'pen': '펜',
            'hat': '모자',
            'mobile_phone': '핸드폰',
            'sock': '양말',
            'glasses': '안경',
            'towel': '수건',
            'newspaper': '신문',
            'remote': '리모컨',
            'key': '열쇠',
            'wallet': '지갑',
            'pack': '담배갑',
            'medicine_case': '약통',
            'medicine_packet': '약봉지',
            'sink': '싱크대',
            'table': '테이블',
            'bed': '침대',
            'sofa_bed': '소파',
            'refrigerator': '냉장고',
            'television': '티비',
            'toilet': '화장실',
            'mechanical_fan': '선풍기',
        }

        # initilize the network here.
        if self.args.net == 'vgg16':
            from model.faster_rcnn.vgg16 import vgg16
            self.fasterRCNN = vgg16(self.classes, pretrained=False, class_agnostic=self.args.class_agnostic)
        elif 'res' in self.args.net:
            # from model.faster_rcnn.resnet import resnet
            if self.args.use_FPN:
                from model.fpn.resnet_AIRvar_CBAM import resnet
            else:
                from model.faster_rcnn.resnet_AIRvar_CBAM import resnet
            if self.args.net == 'res101':
                self.fasterRCNN = resnet(self.classes, 101, pretrained=False, class_agnostic=self.args.class_agnostic, att_type=att_type)
            elif self.args.net == 'res50':
                self.fasterRCNN = resnet(self.classes, 50, pretrained=False, class_agnostic=self.args.class_agnostic, att_type=att_type)
            elif self.args.net == 'res152':
                self.fasterRCNN = resnet(self.classes, 152, pretrained=False, class_agnostic=self.args.class_agnostic, att_type=att_type)
        else:
            print("network is not defined")
            pdb.set_trace()

        self.fasterRCNN.create_architecture()

        print("load checkpoint %s" % (load_name))
        if self.args.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            self.cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')

        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.args.cuda > 0:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        with torch.no_grad():
            self.im_data = Variable(self.im_data)
            self.im_info = Variable(self.im_info)
            self.num_boxes = Variable(self.num_boxes)
            self.gt_boxes = Variable(self.gt_boxes)

        if self.args.cuda > 0:
            self.cfg.CUDA = True

        if self.args.cuda > 0:
            self.fasterRCNN.cuda()

        self.fasterRCNN.eval()

        self.max_per_image = 100

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def detect(self, im_in):
        if len(im_in.shape) == 2:       # if gray == 1 ch
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # im = im_in[:,:,::-1]

        blobs, im_scales = self._get_image_blob(im_in)  # Image in as BGR order
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.resize_(1, 1, 5).zero_()
        self.num_boxes.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if self.cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.args.class_agnostic:
                    if self.args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        ret_bbox_score_class = []    # bbox(4), score(1), class_name(1)
        for j in range(1, len(self.classes)):
            if self.classes[j] in self.display_classes.keys():
                inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)

                print(self.classes[j], inds.numel())
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], self.cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    for k in range(cls_dets.shape[0]):
                        # tensor to numpy
                        ret_bbox_score_class.append([tuple(int(np.round(x.cpu())) for x in cls_dets[k, :4]), cls_dets[k, 4].item(), self.classes[j]])

        # pdb.set_trace()

        return ret_bbox_score_class

    def get_possible_class(self):
        return self.display_classes.keys()

    def visualize(self, image, list_info, box_color=(0, 204, 0), text_color=(255, 255, 255),
                                   text_bg_color=(0, 204, 0), fontsize=20, thresh=0.8, draw_score=True,
                                   draw_text_out_of_box=True, map_classname_to_korean=None):
        """Visual debugging of detections."""
        # print(list_info)

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