# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np

from python_portrait_relight.relight import Relight
from python_portrait_relight._retinaface import RetinaFaceSDK, cfg_mnet

def demo():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    fast = True
    # folder
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    weight_folder = os.path.join(cur_dir, 'weights')
    img_folder = os.path.join(cur_dir, 'imgs')
    # path
    weight_path = os.path.join(weight_folder, 'mobilenet0.25_Final.pth')
    img_num = 6
    img_names = ['portrait_s1.jpg' for _ in range(img_num)]
    ref_names = ['portrait_r{}.jpg'.format(i) for i in range(1, 1+img_num)]
    out_names = ['portrait_o{}.jpg'.format(i) for i in range(1, 1+img_num)]
    # cls init
    RT = Relight(fast=fast)
    RF = RetinaFaceSDK(weight_path=weight_path, cpu=False, cfg=cfg_mnet)
    for img_name, ref_name, out_name in zip(img_names, ref_names, out_names):
        img_path = os.path.join(img_folder, img_name)
        ref_path = os.path.join(img_folder, ref_name)
        out_path = os.path.join(img_folder, out_name)
        # inputs
        img_arr = cv2.imread(img_path)
        [h, w, c] = img_arr.shape
        print('{}: {}x{}x{}'.format(img_name, h, w, c))
        ref_arr = cv2.imread(ref_path)
        [h, w, c] = ref_arr.shape
        print('{}: {}x{}x{}'.format(ref_name, h, w, c))
        # detect
        [boxes, _, _] = RF.detect(img_arr, thre=0.5)
        if len(boxes) == 0:
            print('detect no faces in {}'.format(img_name))
            continue
        [ref_boxes, _, _] = RF.detect(ref_arr, thre=0.5)
        if len(ref_boxes) == 0:
            print('detect no faces in {}'.format(ref_name))
            continue
        # relight
        stime = time.time()
        out_col = RT.relight(img_arr=img_arr, ref_arr=ref_arr,
                             box=boxes[0], ref_box=ref_boxes[0],
                             with_color=True,
                            )
        print('relight time with color: {:.2f}'.format(time.time()-stime))
        stime = time.time()
        out_lig = RT.relight(img_arr=img_arr, ref_arr=ref_arr,
                             box=boxes[0], ref_box=ref_boxes[0],
                             with_color=False,
                            )
        print('relight time with light: {:.2f}\n'.format(time.time()-stime))
        # resize for display
        [ref_height, ref_width, _] = ref_arr.shape
        [height, width, _] = img_arr.shape
        ref_width = int(ref_width*height/ref_height)
        ref_height = height
        ref_arr = cv2.resize(ref_arr, (ref_width, ref_height))
        # save
        cv2.imwrite(out_path, np.concatenate((img_arr, ref_arr, out_col, out_lig), axis=1))

if __name__ == '__main__':
    demo()
