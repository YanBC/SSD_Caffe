import argparse
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', help='path to data directory')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataDir = os.path.abspath(args.dataset)

    dataDir_split = dataDir.split('/')
    dataset_name = dataDir_split[-1] if dataDir_split[-1] != '' else dataDir_split[-2]

    caffe_root = '/opt/caffe/'
    script_dir = os.path.join(caffe_root, 'scripts/create_annoset.py')
    example_dir = os.path.join(os.path.join(caffe_root, 'examples'), dataset_name)

    anno_type = 'detection'
    mapfile = os.path.join(dataDir, 'labelmap.prototxt')
    min_dim = 0
    max_dim = 0
    resize_width = 0
    resize_height = 0
    data_root_dir = os.path.abspath('.')

    for subset in ['test', 'trainval']:
        listfile = os.path.join(dataDir, 'ImageSets/List/', subset + '.txt')
        out_dir = os.path.join(dataDir, 'lmdb', '{}_{}_lmdb'.format(dataset_name, subset))

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