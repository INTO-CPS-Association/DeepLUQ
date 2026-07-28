#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13.06.2024 13:24
# @Author  : Chengjie
# @File    : test_iou.py
# @Software: PyCharm

import unittest
import torch
from torchmetrics.detection import IntersectionOverUnion, DistanceIntersectionOverUnion, CompleteIntersectionOverUnion
from torchmetrics.functional.detection import intersection_over_union


class IoUTest(unittest.TestCase):
    def test_iou(self):
        preds = [{'boxes': torch.tensor([
            [1984.5255, 1571.9885, 2065.0166, 1652.4202],
            [1873.7277, 475.8370, 2007.6564, 625.5668],
            [1917.6455, 774.9953, 2035.7958, 829.4784],
            [1135.3271, 446.3783, 1157.9269, 536.3960],
            [1163.9282, 1622.0948, 1200.7402, 1720.5939]
        ]),
            'scores': torch.tensor([1.0000, 1.0000, 1.0000, 0.8922, 0.7738]),
            'labels': torch.tensor([2, 1, 2, 1, 1]), 'logits': torch.tensor([[8.5148e-04, 5.1892e-09, 1.0000e+00],
                                                                             [8.6015e-06, 1.0000e+00, 8.5848e-10],
                                                                             [8.0912e-04, 7.0934e-06, 1.0000e+00],
                                                                             [1.0513e-03, 8.9220e-01, 1.0199e-03],
                                                                             [6.9864e-04, 7.7382e-01,
                                                                              2.0570e-04]])}]

        targets = [{'boxes': torch.tensor([
            [1984.0438, 1573.7706, 2064.4905, 1655.0498],
            [1876.6918, 475.2428, 2006.8019, 622.8090],
            [1915.1143, 771.9531, 2038.9692, 832.0872],
            [1167.4138, 1622.5602, 1207.1177, 1720.7360],
            [1136.1854, 459.5754, 1153.8789, 531.5005]
        ]),
            'labels': torch.tensor([2, 1, 2, 1, 1]), 'image_id': torch.tensor([30]),
            'area': torch.tensor([5390, 7873, 6051, 3415, 1193]), 'iscrowd': torch.tensor([0, 0, 0, 0, 0]),
            'image_name': 'image_open56.jpg'},
        ]
        metric = IntersectionOverUnion(class_metrics=True, respect_labels=True)
        r = metric(preds, targets)
        # print(r, metric.iou_matrix)

        r = intersection_over_union(preds[0]['boxes'], targets[0]['boxes'], aggregate=False)
        print(r)
