import argparse
import os
import cv2 as cv
from random import shuffle, sample
from shutil import rmtree


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', help='path to data directory')
    p.add_argument('--clean', action='store_true', help='reverse action of calling this script without --clean, namely, delete every file and folder that were to be created by this script')
    return p.parse_args()


def split_data(dataDir):
    print('# Spliting data ...')
    xmlfilepath = os.path.join(dataDir, 'Annotations')
    desDir = os.path.join(dataDir, 'ImageSets/Main/')
    # os.makedirs(desDir, exist_ok=True)
    os.makedirs(desDir)
    ftrainvalPath = os.path.join(desDir, 'trainval.txt')
    ftestPath = os.path.join(desDir, 'test.txt')
    ftrainPath = os.path.join(desDir, 'train.txt')
    fvalPath = os.path.join(desDir, 'val.txt')

    trainval_percent = 0.9
    train_percent = 0.9
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list_range = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = sample(list_range, tv)
    train = sample(trainval, tr)
      
    print("train and val size: {}".format(tv))
    print("train size: {}".format(tr))
    ftrainval = open(ftrainvalPath, 'w')
    ftest = open(ftestPath, 'w')
    ftrain = open(ftrainPath, 'w')
    fval = open(fvalPath, 'w')
      
    for i in list_range:
        name=total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
        
    ftrainval.close()
    ftrain.close() 
    fval.close()  
    ftest .close()


def create_lists(dataDir):
    print('# Creating lists ...')
    splitDir = os.path.join(dataDir, 'ImageSets/Main/')
    ftrainvalPath = os.path.join(splitDir, 'trainval.txt')
    ftestPath = os.path.join(splitDir, 'test.txt')
    ftrainPath = os.path.join(splitDir, 'train.txt')
    fvalPath = os.path.join(splitDir, 'val.txt')

    desDir = os.path.join(dataDir, 'ImageSets/List/')
    # os.makedirs(desDir, exist_ok=True)
    os.makedirs(desDir)

    imageDir = os.path.join(dataDir, 'JPEGImages')
    annoDir = os.path.join(dataDir, 'Annotations')

    # create trainval.txt
    with open(os.path.join(desDir, 'trainval.txt'), 'w') as des:
        with open(ftrainvalPath) as src:
            dataNames = src.readlines()

        shuffled_dataNames = [x.strip() for x in dataNames]
        shuffle(shuffled_dataNames)

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
            print('{} does not exist. Continue ...'.format(imagePath))
            continue
        height, width, _ = image.shape

        fdestest.write('{} {}\n'.format(imagePath, annoPath))
        fdestestnamesize.write('{} {} {}\n'.format(name, height, width))

    fdestest.close()
    fdestestnamesize.close()


def create_data(dataDir, dataset_name, caffe_root='/opt/caffe'):
    print('# Creating lmdb ...')
    datasetDir = os.path.abspath(dataDir)

    script_dir = os.path.join(caffe_root, 'scripts/create_annoset.py')
    example_dir = os.path.join(os.path.join(caffe_root, 'examples'), dataset_name)

    anno_type = 'detection'
    mapfile = os.path.join(datasetDir, 'labelmap.prototxt')
    min_dim = 0
    max_dim = 0
    resize_width = 0
    resize_height = 0
    data_root_dir = os.path.abspath('.')

    for subset in ['test', 'trainval']:
        listfile = os.path.join(datasetDir, 'ImageSets/List/', subset + '.txt')
        out_dir = os.path.join(datasetDir, 'lmdb', '{}_{}_lmdb'.format(dataset_name, subset))

        os.system('python {} \
                    --anno-type={} \
                    --label-map-file={} \
                    --min-dim={} \
                    --max-dim={} \
                    --resize-width={} \
                    --resize-height={} \
                    --check-label \
                    --encode-type=jpg \
                    --encoded \
                    --redo \
                    {} {} {} {}'.format(script_dir, anno_type, mapfile, min_dim, max_dim, resize_width, resize_height, data_root_dir, listfile, out_dir, example_dir))




if __name__ == '__main__':
    args = get_args()
    dataDir = args.dataset
    dataDir_split = dataDir.split('/')
    dataName = dataDir_split[-1] if dataDir_split[-1] != '' else dataDir_split[-2]
    caffe = '/opt/caffe/'

    if args.clean:
        rmtree(os.path.join(dataDir, 'ImageSets/'), ignore_errors=True)
        rmtree(os.path.join(dataDir, 'lmdb'), ignore_errors=True)
        rmtree(os.path.join(caffe, 'examples', dataName), ignore_errors=True)
    else:
        split_data(dataDir)
        create_lists(dataDir)
        create_data(dataDir, dataName, caffe_root=caffe)

