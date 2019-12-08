import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
from keras.models import load_model
import sys
import csv


# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self, is_site):
        
        self.is_site = is_site
        cwd = os.path.dirname(os.path.realpath(__file__))

        # load keras Lenet style model from file
        if self.is_site is True:
            self.class_model = load_model(cwd+'/model/model.h5')
        else:
            self.class_model = load_model(cwd+'/model/model_sim.h5')
        self.class_graph = tf.get_default_graph()

        self.dg = self.load_graph(cwd+"/model/frozen_inference_graph.pb")
        
        #get names of nodes. from https://www.activestate.com/blog/2017/08/using-pre-trained-models-tensorflow-go
        self.session = tf.Session(graph=self.dg )
        self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
        self.detection_boxes =  self.dg.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
        self.num_detections    = self.dg.get_tensor_by_name('num_detections:0')

        self.tlclasses = [ TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN ]
        self.tlclasses_d = { TrafficLight.RED: "RED", TrafficLight.YELLOW:"YELLOW", TrafficLight.GREEN:"GREEN", TrafficLight.UNKNOWN:"UNKNOWN" }
#        self.tlclasses = [ 0, 1, 2 ]
#        self.tlclasses_d = { 0 : "RED", 1:"YELLOW", 2:"GREEN", -1:"UNKNOWN" }

        pass

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and classes[i] == 10:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

            
    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light. OpenCV is BGR by default.
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        box = self.localize_lights( image )
        if box is None:
            return TrafficLight.UNKNOWN
        class_image = cv2.resize( image[box[0]:box[2], box[1]:box[3]], (32,32) )
        return self.classify_lights( class_image )



    def classify_lights(self, image):
        """ Given a 32x32x3 image classifies it as red, greed or yellow
            Expects images in BGR format. Important otherwide won't classify correctly
            
        """
        status = TrafficLight.UNKNOWN
        img_resize = np.expand_dims(image, axis=0).astype('float32')
        with self.class_graph.as_default():
            predict = self.class_model.predict(img_resize)
            status  = self.tlclasses[ np.argmax(predict) ]

        return status #np.argmax(predict)



    def localize_lights(self, image):
        """ Localizes bounding boxes for lights using pretrained TF model
            expects BGR8 image
        """

        with self.dg.as_default():
            #switch from RGB to BGR. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)


            ret = None

            confidence_cutoff = 0.3
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, detection_boxes, detection_scores, detection_classes)


            dim = image.shape[0:2]
            for idx in range(len(boxes)):
                box = self.from_normalized_dims__to_pixel(boxes[idx], dim)
                box_h, box_w  = (box[2] - box[0], box[3]-box[1] )
                if (box_h <20) or (box_w<20):  
                    pass    # box too small 
                elif ( box_h/box_w <1.2):
                    pass    # wrong ratio
                else:
                    #rospy.loginfo('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                    ret = box
                    break

        return ret
        
    def from_normalized_dims__to_pixel(self, box, dim):
            height, width = dim[0], dim[1]
            box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
            return np.array(box_pixel)


    def draw_box(self, img, box):
        cv2.rectangle(img, (box[1],box[0]), (box[3],box[2]), (255,0,0), 5)
        return img



if __name__ == '__main__':
    if False:
        # test localizatoin
        cl = TLClassifier()
        #img = np.asarray( Image.open('images/3.jpg'), dtype="uint8" )
        img = cv2.imread('images/ts-1251.jpg')
        box = cl.localize_lights( img )
        if(box is None):
            print("Couldn't locate lights")
        else:
            print(box)
            crop = img[box[0]:box[0]+(box[2]-box[0]), box[1]:box[1]+(box[3]-box[1])]
            dst = cv2.resize( crop, (32, 32) )
            cv2.imshow("cropped", crop)
            cv2.waitKey(0)
            cv2.imwrite("images/out2.jpg", dst)
            cv2.imwrite("images/out.jpg", cl.draw_box( img, box))

    if False:
        cl = TLClassifier()
        LABEL_FILE = 'sim_image_data_and_label/traffic_labels.txt'
        with open(LABEL_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                img = cv2.imread(os.path.join('sim_image_data_and_label',row[0]))
                box = cl.localize_lights( img )
                if(box is None):
                    print("Couldn't locate lights")
                else:
                    crop = img[box[0]:box[0]+(box[2]-box[0]), box[1]:box[1]+(box[3]-box[1])]
                    dst = cv2.resize( crop, (32, 32) )
                    path = None
                    if int(row[1]) == 0:
                        path = 'sim_image_data_and_label/Red'
                    elif int(row[1]) == 1:
                        path = 'sim_image_data_and_label/Yellow'
                    elif int(row[1]) == 2:
                        path = 'sim_image_data_and_label/Green'
                    else:
                        path = None
                    if path is not None:
                        cv2.imwrite(os.path.join(path,row[0]), dst)
                    cv2.imshow("Labeled", cl.draw_box( img, box))
                    cv2.waitKey(10)


    if False:
        #preprocess training images. produce 32x32 images that don't contain background
        for i in range(3):
            paths = glob(os.path.join('classifier_images/labeled_original/{}'.format(i), '*.png'))
            for path in paths:
                print(path)
                img = cv2.imread(path)
                crop = img[3:29, 11:22]
                dst = cv2.resize( crop, (32, 32) )
                cv2.imwrite("prep/"+path, dst)

    if False:
        cl = TLClassifier()
        input_yaml = sys.argv[1]
        images = get_all_labels(input_yaml)
        paths = glob(os.path.join('images/', '*.png'))
        for path in paths:
            img = cv2.imread(path)
            status = cl.get_classification( img )
            print( cl.tlclasses_d[status], path)
            # print( status )
        myList = ['Red','Green','Yellow']
        for i, image_dict in enumerate(images):
            print(image_dict['path'])
            for box in image_dict['boxes']:
                if box['label'] in myList:
                    image = cv2.imread(image_dict['path'])
                    status = cl.get_classification( image )
                    print( cl.tlclasses_d[status], ',', box['label'])

    if False:
        cl = TLClassifier(False)
        # input_yaml = sys.argv[1]
        # images = get_all_labels(input_yaml)
        paths = glob(os.path.join('test/', '*.jpg'))
        for path in paths:
            img = cv2.imread(path)
            status = cl.get_classification( img )
            print( cl.tlclasses_d[status], path)
            # print( status )
        # myList = ['Red','Green','Yellow']
        # for i, image_dict in enumerate(images):
        #     print(image_dict['path'])
        #     for box in image_dict['boxes']:
        #         if box['label'] in myList:
        #             image = cv2.imread(image_dict['path'])
        #             status = cl.get_classification( image )
        #             print( cl.tlclasses_d[status], ',', box['label'])

