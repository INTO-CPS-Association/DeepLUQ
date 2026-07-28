#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/17 下午4:15
# @Author  : Chengjie Lu
# @File    : test_clustering.py
# @Software: PyCharm

import unittest
from src.deepluq.utils import wbf_clustering


class IoUTest(unittest.TestCase):
    def test_wbf_clustering(self):
        # --- Example Execution ---

        # Note: Using your provided JSON structure logic
        your_predictions = {
            'pred_0': {'box': [822.0, 301.6, 902.1, 340.7], 'box_n': [0.642, 0.419, 0.704, 0.473], 'label': 0,
                       'score': 0.932, 'logit': [0.932, 1.107e-08, 0.0002]},
            'pred_1': {'box': [865.9, 309.1, 892.9, 332.6], 'box_n': [0.676, 0.429, 0.697, 0.462], 'label': 2,
                       'score': 0.877, 'logit': [0.0005, 7.545e-06, 0.877]},
            'pred_2': {'box': [822.0, 301.6, 902.1, 340.7], 'box_n': [0.642, 0.419, 0.704, 0.473], 'label': 0,
                       'score': 0.900, 'logit': [0.900, 2.107e-08, 0.0003]},
        }

        import json
        grouped_results = wbf_clustering(your_predictions, iou_thr=0.5, skip_box_thr=0.01)
        print(json.dumps(grouped_results, indent=4))
