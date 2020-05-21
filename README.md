# Setup
1. Put your dataset under `datasets` folder, your dataset should be of voc format and contains these folders and file: `Annotations`, `JPEGImages`, `labelmap.prototxt`

2. Create the lmdb database by running the following cmd
```bash
cd <project-root>
python ./datasets/create_lmdb.py <path-to-your-dataset>
```

3. Put your caffe model files under `models` folder, you should define the following files:
- `train.prototxt`: model definition for training
- `test.prototxt`: model definition for testing
- `deploy.prototxt`: model definition for deployment
- `solver.prototxt`: training solver definition



# Train
Train command example:
```bash
caffe train \
    -solver models/Modified_ResNet_SSD_voc_hand_plus_egohands/solver.prototxt \
    -weights models/Modified_ResNet_SSD_voc_hand_plus_egohands/resnet_ssd_iter_5000.caffemodel
```



# Test
Test command example:
```bash
python ssd_detect_video.py \
        datasets/voc_hand/labelmap.prototxt \
        models/VGGNet_SSD/SSD_300x300/deploy.prototxt \
        models/VGGNet_SSD/snapshots/vgg_ssd_iter_50000.caffemodel \
        /workspace/hand_detection/videos/VID_20200423_103204.mp4 \
        .
```




# Eval
Eval command example:
```bash
python ssd_eval.py \
        datasets/voc_hand_plus_egohands/ImageSets/List/test.txt \
        datasets/voc_hand_plus_egohands/labelmap.prototxt \
        models/ResNet_SSD/ResNet_SSD_deploy.prototxt \
        models/ResNet_SSD/snapshots/resnet_ssd_iter_50000.caffemodel \
        --iou_thres 0.5 \
        --gpu_id 1 \
        --cache ./cache/ResNet_SSD.pkl
```