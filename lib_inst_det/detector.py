import _init_paths
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cv2

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from model.utils.config import cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import numpy as np
import argparse
import pprint

# every complex things are in this class
# user needs just to call the APIs.
# detector - faster-rcnn.pytorch
class Detector():
    def __init__(self, baseFolder='models', threshold=0.9):
        super(Detector, self).__init__()

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

            return parser

        cmd_args = [
            '--net', 'res101',
            '--ls',
            '--cuda',
        ]

        # load_name = 'output/MSCOCO/res101/faster_rcnn_1_10_14657.pth'
        load_name = os.path.join(baseFolder, 'faster_rcnn_1_10_14657.pth')

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
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ])

        # self.display_classes = self.classes
        self.display_classes = [
            'bottle', 'cup', 'cell phone', 'remote'
        ]

        # initilize the network here.
        if self.args.net == 'vgg16':
            self.fasterRCNN = vgg16(self.classes, pretrained=False, class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res101':
            self.fasterRCNN = resnet(self.classes, 101, pretrained=False, class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res50':
            self.fasterRCNN = resnet(self.classes, 50, pretrained=False, class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res152':
            self.fasterRCNN = resnet(self.classes, 152, pretrained=False, class_agnostic=self.args.class_agnostic)
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

        self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.data.resize_(1, 1, 5).zero_()
        self.num_boxes.data.resize_(1).zero_()

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
            if self.classes[j] in self.display_classes:
                inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, self.cfg.TEST.NMS, force_cpu=not self.cfg.USE_GPU_NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    for k in range(cls_dets.shape[0]):
                        # tensor to numpy
                        ret_bbox_score_class.append([tuple(int(np.round(x)) for x in cls_dets[k, :4]), cls_dets[k, 4].item(), self.classes[j]])

        return ret_bbox_score_class

    def get_possible_class(self):
        return self.display_classes
