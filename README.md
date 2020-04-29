# Train
```bash
# you should first enter a caffe container
# and cd to project root
cd ./datasets
mkdir -p ./VOC2007/ImageSets/Main
python train_val.py
cd ..

./create_list.sh
mv test.txt test_name_size.txt trainval.txt /opt/caffe/data/VOC2007/
cp labelmap_voc.prototxt /opt/caffe/data/VOC2007/

./create_data.sh 
./train.sh
```