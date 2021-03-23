# -*- coding: utf-8 -*-
import numpy as np
import random
def generate_comment(person_bbox, list_classes, list_objs_bbox, context):
    # person_bbox: [x1, y1, x2, y2]
    # list_classes: list of objects' indexes detected by detector
    # list_objs_bbox: list of bboxes
    # context: 0 : undefined,1 : in the morning, 2 : in the noon,3 : in the night, 4 : when go out (외출할 때), 5 : when come in (외출후 들어올 때), 9: first seeing (처음 본 사람)
    # int gender: 0: undefined, 1: male, 2: female

    def calculate_iou(bboxA, bboxB):
        inter_x1 = np.maximum(bboxA[:2], bboxB[:2])
        inter_x2 = np.minimum(bboxA[2:], bboxB[2:])

        inter_area = max(inter_x2[0] - inter_x1[0], 0) * max(inter_x2[1] - inter_x1[1], 0)
        bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
        bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])

        iou = float(inter_area) / float(bboxA_area + bboxB_area - inter_area)

        return iou

    # 0. predefined variables
    list_obj_and_comment = {'cell phone': {4: '핸드폰을 들고 다니시면 떨어뜨릴 수 있어요. 주머니에 넣고 다니세요.',
                                           5: '핸드폰을 들고 다니시면 떨어뜨릴 수 있어요. 주머니에 넣고 다니세요.'},
                            'tie': {4: '중요한 자리에 가시나봐요. 안녕히 다녀오세요.',
                                    5: '중요한 자리, 잘 다녀오셨나요? 오늘 하루도 고생하셨습니다.'},
                            'umbrella': {4: '우산을 가져가시네요. 비오는 중에는 앞을 꼭 보며 걸으셔야해요.',
                                         5: '비오는 중에, 안녕히 다녀오셨나요?'}
                              }

    # 1. first check the position and name of object bboxes
    list_candidate_comments = []
    for iobj, obj_bbox in enumerate(list_objs_bbox):
        obj_cx = (obj_bbox[0] + obj_bbox[2]) / 2
        obj_cy = (obj_bbox[1] + obj_bbox[3]) / 2

        # check object's center is in person_bbox
        if obj_cx > person_bbox[0] and obj_cx < person_bbox[2] and obj_cy > person_bbox[1] and obj_cy < person_bbox[3]:

            # check object is in comment_object_list:
            try:
                list_candidate_comments.append(list_obj_and_comment[list_classes[iobj]][context])
            except:
                pass

    # 2. randomly select one comment
    ret_comment = None
    if len(list_candidate_comments) > 0:
        ret_comment = random.choice(list_candidate_comments)

    return ret_comment


if __name__ == '__main__':
    # detected persons' bbox
    person_bbox = [10, 10, 40, 70]

    # detected objects' class name: according to COCO
    list_classes = ['tie',      # target object
                    'cup',      # not in comment list
                    'umbrella'  # not in person's bbox
                    ]
    list_objs_bbox = [
        [20, 15, 25, 20],
        [20, 15, 25, 20],
        [50, 15, 70, 20]
    ]

    comment = generate_comment(person_bbox, list_classes, list_objs_bbox, 5)

    print(comment)
