from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

PIXEL_BGR_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

class det_to_cls(nn.Module):
    def __init__(self, base, top, clf, roi_align):
        super(det_to_cls, self).__init__()

        self.base = base
        self.top = top
        self.clf = clf
        self.roi_align = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, input):
        pool5 = self.base(input)    # 1, 3, 224, 224 -> 1, 1024, 14, 14
        pool5 = self.roi_align(pool5)   # -> 1, 1024, 7, 7

        fc7 = self.top(pool5).mean(3).mean(2)   # -> 1, 2048
        score = self.clf(fc7)       # -> 1, n_class+1

        mask = torch.ones(size=score.shape, dtype=score.dtype)
        mask[:, 0] = 0

        score = score * mask.cuda()
        cls_prob = F.softmax(score, 1)

        return cls_prob

class gradCam():
    def __init__(self, model):
        super(gradCam, self).__init__()

        self.model = det_to_cls(model.fasterRCNN.RCNN_base,
                                model.fasterRCNN.RCNN_top,
                                model.fasterRCNN.RCNN_cls_score,
                                model.fasterRCNN.RCNN_roi_align)

        self.target_layer = self.model.top[-1]

        # self.cam = GradCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)
        # self.cam = GradCAMPlusPlus(model=self.model, target_layer=self.target_layer, use_cuda=True)
        # self.cam = ScoreCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)        # too long time
        self.cam = XGradCAM(model=self.model, target_layer=self.target_layer, use_cuda=True)

    def get_gradCam(self, input, target_id):
        grayscale_cam = self.cam(input_tensor=input, target_category=target_id)

        return grayscale_cam

    def get_score(self, input):
        score = self.model(input)

        return score

    def convert_to_tensor(self, input_bgr):
        input_bgr = input_bgr - PIXEL_BGR_MEANS

        im_data_pt = torch.from_numpy(input_bgr)
        im_data_pt.unsqueeze_(0)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_data_pt = im_data_pt.cuda().float()

        return im_data_pt