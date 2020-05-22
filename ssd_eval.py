# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2 as cv
import pickle
import argparse
from nn import CaffeDetection
import xml.etree.ElementTree as ET
from google.protobuf import text_format
from caffe.proto import caffe_pb2



def _IoU(boxA, boxB):
    '''
    Calculate the intersection over union ratio between two boxes
    '''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def _findMaxIoU(target, gt_list):
    '''
    Find the box that has the maximun IoU with target

    INPUT:
        target <list>: coordinates of the target box in the form of
                            [left, top, right, bottom]

        gt_list <list>: a list of boxes that have the same form of
                            target

    OUTPUT:
        max_iou <float>: the found maximum IoU between target box 
                            and boxes in gt_list

        gt_index <int>: the index of the box in gt_list that has
                            the maximum IoU with target
    '''
    max_iou = 0
    gt_index = -1
    for i in range(len(gt_list)):
        tmp_gt = gt_list[i]
        tmp_iou = _IoU(tmp_gt, predition)
        if tmp_iou > max_iou:
            max_iou = tmp_iou
            gt_index = i

    return gt_index, max_iou        # return -1,0 if not found


def _maxIoUSuppression(iou_mat, iouThres=0.5):
    '''
    Assign each predicted box to ground true boxes based on IoU

    INPUT:
        iou_mat <2d array>: IoU matrix. Entry_i,j is the IoU ratio 
                            of the i_th ground true box and the 
                            j_th predicted box

        iouThres <float>: IoU threshold. Any predicted box that has
                            an IoU lower that this threshold will 
                            be considered false positive

    OUTPUT:
        gt_indices <1d array>: a 1d array of length iou_mat.shape[1],
                                gt_indices[j] is the index of ground
                                true box assigned to predicted box j;
                                gt_indices[j] = -1 if no ground true
                                box is assigned
    '''
    gt_indices = np.argmax(iou_mat, axis=0)

    for j in range(len(gt_indices)):
        gt_index = gt_indices[j]

        # if iou(gt, pd) is less than iouThres,
        # the corresponding pd is a false positive
        if iou_mat[gt_index, j] < iouThres:
            gt_indices[j] = -1
            continue

        # check if current prediction is matched with a 
        # ground-true box that has already been assigned
        if gt_index in set(gt_indices[:j]):
            prev_j = np.where(gt_indices[:j]==gt_index)[0][0]
            if iou_mat[gt_index, prev_j] > iou_mat[gt_index, j]:
                gt_indices[j] = -1
            else:
                gt_indices[prev_j] = -1

    return gt_indices


def proto2dict(labelmap):
    ret = dict()
    for item in labelmap.item:
        ret[item.display_name] = item.label
    return ret


def getPredictions(model, imageList):
    '''
    make predictions on all images in imageList

    INPUT:
        model <CaffeDetection>: the to-be-evaluated model 

        imageList <list>: paths to images


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file path
                          
                          Value <list>: 
                            annotations in the form of [anno1, anno2, ..., annoN]
                                
                            anno <tuple>:
                                (confidence <float>, class <int>, 
                                    [left <int>, top <int>, right <int>, bottom <int>])
    '''
    ret = dict()
    for imagePath in imageList:
        img = cv.imread(imagePath)

        tmp = []
        boxes = model.detect(img)
        for box in boxes:
            confidence = float(box[5])
            classId = int(box[4])
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])

            tmp.append((confidence, classId, [left, top, right, bottom]))
        ret[imagePath] = tmp

    return ret


def getAnnotations(listFile, labelDict):
    '''
    Read annotations

    INPUT:
        dirPath <string>: path to image and annotation directory


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file name
                          
                          Value <list>: 
                            annotations in the form of [anno1, anno2, ..., annoN]
                                
                            anno <tuple>:
                                (class <int>, [left <int>, top <int>, right <int>, bottom <int>])
    '''
    with open(listFile) as f:
        lines = f.readlines()

    ret = dict()
    for line in lines:
        imagePath, annoPath = line.strip().split(' ')

        try:
            annoRoot = ET.parse(annoPath).getroot()
        except IOError:
            print('{} does not exit'.format(annoPath))
            continue

        objects = []
        for obj in annoRoot.findall('object'):
            label = labelDict[obj.find('name').text]
            left = int(obj.find('bndbox/xmin').text)
            top = int(obj.find('bndbox/ymin').text)
            right = int(obj.find('bndbox/xmax').text)
            bottom = int(obj.find('bndbox/ymax').text)
            objects.append((label, [left, top, right, bottom]))

        ret[imagePath] = objects

    return ret


def getStats(classId, y_pred, y_true, iouThres=0.5):
    '''
    Perform statistical analysis on bounding box predictions

    INPUT:
        classId <int>: the class to be analyzed

        y_pred <dictionary>: predictions; output from getPredictions()

        y_true <dictionary>: ground-true; output from getAnnotations()

        iouThres <float>: IoU threshold for positive predictions

    OUTPUT:
        tp_size <int>: number of true positives

        fp_size <int>: number of false positves

        fn_size <int>: number of false negatives

        ret_tp <list>: true positives information. each element is a true 
                        positive's confidence and its IoU with groud true:
                            [confidence, IoU]
    '''
    fn_size = 0
    fp_size = 0
    tp_size = 0
    ret_tp = []

    # make sure images in y_pred are the same with images in y_true
    y_pred_images = sorted(list(y_pred.keys()))
    y_true_images = sorted(list(y_true.keys()))
    assert y_pred_images == y_true_images
    images = y_pred_images

    # loop over each image
    for image in images:
        gt_list = [box[1] for box in y_true[image] if box[0] == classId]
        pd_list = [box[2] for box in y_pred[image] if box[1] == classId]

        # if there is no predicted boxes, all ground true
        # boxes would go to the false negative category
        if len(pd_list) == 0:
            fn_size += len(gt_list)
            continue

        # if there is no ground true boxes, all predicted 
        # boxes would be false negatives
        if len(gt_list) == 0:
            fp_size += len(pd_list)
            continue

        # get iou matrix
        iou = np.empty((len(gt_list), len(pd_list)))
        for i in range(len(gt_list)):
            for j in range(len(pd_list)):
                iou[i,j] = _IoU(gt_list[i], pd_list[j])

        # decide whether a predicted box is true positive 
        # or false positive
        gt_assigned = _maxIoUSuppression(iou, iouThres)

        # record all true positives to ret_tp
        preds = np.where(gt_assigned != -1)[0]
        for pred in preds:
            tmp_confidence = y_pred[image][pred][0]
            tmp_iou = iou[gt_assigned[pred], pred]

            ret_tp.append([tmp_confidence, tmp_iou])

        # record false positives and false negatives
        fp_size += len(pd_list) - len(preds)
        fn_size += len(gt_list) - len(preds)

    # yes, I am that obsessive-compulsive
    tp_size = len(ret_tp)
    fp_size = int(fp_size)
    fn_size = int(fn_size)
    return tp_size, fp_size, fn_size, ret_tp


def getArgs():
    p = argparse.ArgumentParser(description='Detection Evaluation')
    p.add_argument('test_data_list', help='Path to test list file (test.txt)')
    p.add_argument('labelmap_file', help='path to labelmap file')
    p.add_argument('model_def', help='path to model definition')
    p.add_argument('model_weights', help='path to model weight')
    p.add_argument('--iou_thres', type=float, default=0.5, help='IoU Threshold. Default: 0.5')
    p.add_argument('--gpu_id', type=int, default=0, help='gpu id. Default: 0')
    p.add_argument('--cache', help='path to prediction cache file')
    p.add_argument('--image_resize', type=int, default=300, help='nn input image shape. Default: 300')
    return p.parse_args()



if __name__ == '__main__':
    args = getArgs()
    test_data_list = args.test_data_list
    labelmap_file = args.labelmap_file
    model_def = args.model_def
    model_weights = args.model_weights
    iou_thres = args.iou_thres
    gpu_id = args.gpu_id
    cache_path = args.cache
    image_resize = args.image_resize

    if cache_path is not None and os.path.isfile(cache_path):
        print('# LOADING PREVIOUS DATA ...')
        with open(cache_path, 'rb') as f:
            labelDict, annos, preds = pickle.load(f)
        print('# DONE!\n')

    else:
        # get annotations
        print('# READING ANNOTATIONS ...')
        with open(labelmap_file, 'r') as file:
            labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(file.read()), labelmap)
        labelDict = proto2dict(labelmap)
        annos = getAnnotations(test_data_list, labelDict)
        print('# DONE!\n')

        # get predictions
        print('# MAKING PREDICTIONS ...')
        net = CaffeDetection(model_def, model_weights, labelmap_file, gpu_id, image_resize)
        images = list(annos.keys())
        preds = getPredictions(net, images)
        print('# DONE!\n')

        # write to cach
        if cache_path is not None:
            cache_dir = os.path.dirname(cache_path)
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_path, 'wb') as f:
                pickle.dump((labelDict, annos, preds), f)

    # performance analysis
    print('# ANALYSIS RESULTS')
    print('Using IoU threshold of %0.2f\n' % (iou_thres))

    classNames = list(labelDict.keys())

    for className in classNames:
        c = labelDict[className]
        tps, fps, fns, tp_info = getStats(c, preds, annos, iou_thres)
        tmp = np.array(tp_info)
        try:
            assert tmp.shape[1] == 2
        except IndexError:
            continue

        tmp_avgs = tmp.mean(axis=0)

        print('Class: %s' % className)
        print('Recall: %0.3f' % (float(tps)/float(tps+fns)))
        print('Precision: %0.3f' % (float(tps)/float(tps+fps)))
        print('True Positive:')
        print('    Average Confidence: %0.3f' % (float(tmp_avgs[0])))
        print('    Average IoU: %0.3f\n' % (float(tmp_avgs[1])))


