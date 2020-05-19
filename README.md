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
caffe train -solver models/Modified_ResNet_SSD_voc_hand_plus_egohands/solver.prototxt -weights models/Modified_ResNet_SSD_voc_hand_plus_egohands/resnet_ssd_iter_5000.caffemodel
```



# Test
Test command example:
```bash
python ssd_detect_video.py --gpu_id 1 --labelmap_file labelmap_voc.prototxt --model_def models/ResNet/SSD_Res10_300x300/deploy_3x3.prototxt --model_weights models/ResNet/SSD_Res10_300x300/ResNet10_3x3_iter_54538.caffemodel --image_file /workspace/hand_detection/videos/VID_20200426_135303.mp4
```