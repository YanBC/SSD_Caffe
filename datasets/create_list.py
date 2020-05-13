import argparse
import os
import cv2 as cv
import random


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', help='path to data directory')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataDir = args.dataset

    splitDir = os.path.join(dataDir, 'ImageSets/Main/')
    ftrainvalPath = os.path.join(splitDir, 'trainval.txt')
    ftestPath = os.path.join(splitDir, 'test.txt')
    ftrainPath = os.path.join(splitDir, 'train.txt')
    fvalPath = os.path.join(splitDir, 'val.txt')

    desDir = os.path.join(dataDir, 'lmdb')
    os.makedirs(desDir, exist_ok=True)

    imageDir = os.path.join(dataDir, 'JPEGImages')
    annoDir = os.path.join(dataDir, 'Annotations')

    # create trainval.txt
    with open(os.path.join(desDir, 'trainval.txt'), 'w') as des:
        with open(ftrainvalPath) as src:
            dataNames = src.readlines()

        shuffled_dataNames = [x.strip() for x in dataNames]
        random.shuffle(shuffled_dataNames)

        for name in shuffled_dataNames:
            imagePath = os.path.join(imageDir, name + '.jpg')
            annoPath = os.path.join(annoDir, name + '.xml')
            des.write('{} {}\n'.format(imagePath, annoPath))

    # create test.txt and test_name_size.txt
    fdestest = open(os.path.join(desDir, 'test.txt'), 'w')
    fdestestnamesize = open(os.path.join(desDir, 'test_name_size.txt'), 'w')
    with open(ftestPath) as src:
        dataNames = src.readlines()
    testNames = [x.strip() for x in dataNames]

    for name in testNames:
        imagePath = os.path.join(imageDir, name + '.jpg')
        annoPath = os.path.join(annoDir, name + '.xml')

        image = cv.imread(imagePath)
        if image is None:
            print(f'{imagePath} does not exist. Continue ...')
            continue
        height, width, _ = image.shape

        fdestest.write('{} {}\n'.format(imagePath, annoPath))
        fdestestnamesize.write('{} {} {}\n'.format(name, height, width))

    fdestest.close()
    fdestestnamesize.close()
