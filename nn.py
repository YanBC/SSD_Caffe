import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import numpy as np


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


def clip(x, min_thres, max_thres):
    assert min_thres <= max_thres
    if x <= min_thres:
        return min_thres
    elif x <= max_thres:
        return x
    else:
        return max_thres



class CaffeDetection:
    def __init__(self, model_def, model_weights, labelmap_file, gpu_id, image_resize=300):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        with open(labelmap_file, 'r') as file:
            self.labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(file.read()), self.labelmap)


    def detect(self, image, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        image_height, image_width, _ = image.shape

        # set net to batch size of 1
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        #image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            # xmin = top_xmin[i]
            # ymin = top_ymin[i]
            # xmax = top_xmax[i]
            # ymax = top_ymax[i]

            xmin = int(clip(round(top_xmin[i] * image_width), 0, image_width))
            ymin = int(clip(round(top_ymin[i] * image_height), 0, image_height))
            xmax = int(clip(round(top_xmax[i] * image_width), 0, image_width))
            ymax = int(clip(round(top_ymax[i] * image_height), 0, image_height))

            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])

        return result


