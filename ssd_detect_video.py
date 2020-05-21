#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import cv2 as cv

from nn import CaffeDetection


def main(args):
    '''main '''
    detection = CaffeDetection(args.model_def, args.model_weights, args.labelmap_file, args.gpu_id, args.image_resize)

    srcPath = args.src
    if not os.path.isfile(srcPath):
        print('{} does not exit'.format(srcPath))
    desPath = os.path.join(args.desDir, 'res_{}'.format(os.path.basename(srcPath)))

    video = cv.VideoCapture(srcPath)
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    if num_frames == 1:
        _, frame1 = video.read()
        frame = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame = frame / 255.
        result = detection.detect(frame)

        for item in result:
            xmin, ymin, xmax, ymax = item[0:4]
            cv.rectangle(frame1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv.imwrite(desPath, frame1)

    elif num_frames > 1:
        fps = video.get(cv.CAP_PROP_FPS)
        size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        video_w = cv.VideoWriter(desPath, cv.VideoWriter_fourcc('X','V','I','D'), fps, size)

        while video.isOpened():
            success, frame1 = video.read() 
            if(success == False):
                break;

            frame = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            frame = frame / 255.
            result = detection.detect(frame)

            for item in result:
                xmin, ymin, xmax, ymax = item[0:4]
                cv.rectangle(frame1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            video_w.write(frame1)


def get_args():
    '''parse args'''
    p = argparse.ArgumentParser()
    p.add_argument('labelmap_file', help='path to labelmap file')
    p.add_argument('model_def', help='path to model definition')
    p.add_argument('model_weights', help='path to model weight')
    p.add_argument('src', help='path to video or image file')
    p.add_argument('desDir', help='where to save the result')
    p.add_argument('--image_resize', default=300, type=int, help='resize image to specific shape. Default: 300')
    p.add_argument('--gpu_id', type=int, default=0, help='gpu id. Default: 0')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
