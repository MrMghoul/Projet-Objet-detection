# encapsulation de l'algo YOLO dans une classe
# basé sur https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from keras.models import load_model

import cv2

class BoundBox:	
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score
	
class BaseYOLO:
	# La classe BaseYOLO ne peut pas être instanciée directement
	def __init__(self):
		raise NotImplementedError("BaseYOLO cannot be instantiated directly")

	    
	def _sigmoid(self,x):
		return 1. / (1. + np.exp(-x))

	def decode_netout(self,netout, anchors, obj_thresh, net_h, net_w):
		grid_h, grid_w = netout.shape[:2]
		nb_box = 3
		netout = netout.reshape((grid_h, grid_w, nb_box, -1))
		nb_class = netout.shape[-1] - 5
		boxes = []
		netout[..., :2]  = self._sigmoid(netout[..., :2])
		netout[..., 4:]  = self._sigmoid(netout[..., 4:])
		netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
		netout[..., 5:] *= netout[..., 5:] > obj_thresh

		for i in range(grid_h*grid_w):
			row = i / grid_w
			col = i % grid_w
			for b in range(nb_box):
				# 4th element is objectness score
				objectness = netout[int(row)][int(col)][b][4]
				if(objectness.all() <= obj_thresh): continue
				# first 4 elements are x, y, w, and h
				x, y, w, h = netout[int(row)][int(col)][b][:4]
				x = (col + x) / grid_w # center position, unit: image width
				y = (row + y) / grid_h # center position, unit: image height
				w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
				h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
				# last elements are class probabilities
				classes = netout[int(row)][col][b][5:]
				box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
				boxes.append(box)
		return boxes

	def correct_yolo_boxes(self,boxes, image_h, image_w, net_h, net_w):
		new_w, new_h = net_w, net_h
		for i in range(len(boxes)):
			x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
			y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
			boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
			boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
			boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
			boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

	def _interval_overlap(self,interval_a, interval_b):
		x1, x2 = interval_a
		x3, x4 = interval_b
		if x3 < x1:
			if x4 < x1:
				return 0
			else:
				return min(x2,x4) - x1
		else:
			if x2 < x3:
				return 0
			else:
				return min(x2,x4) - x3

	def bbox_iou(self,box1, box2):
		intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
		intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
		intersect = intersect_w * intersect_h
		w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
		w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
		union = w1*h1 + w2*h2 - intersect
		return float(intersect) / union

	def do_nms(self,boxes, nms_thresh):
		if len(boxes) > 0:
			nb_class = len(boxes[0].classes)
		else:
			return
		for c in range(nb_class):
			sorted_indices = np.argsort([-box.classes[c] for box in boxes])
			for i in range(len(sorted_indices)):
				index_i = sorted_indices[i]
				if boxes[index_i].classes[c] == 0: continue
				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]
					if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
						boxes[index_j].classes[c] = 0

	# load and prepare an image
	def load_image_pixels(self,image, shape):
		
		if not isinstance(image, np.ndarray):
			raise Exception("L'image doit être un tableau numpy")
		
		height, width, channels = image.shape
		# load the image with the required size
		# Retaillage de l'image
		yolo_image = image.copy()
		yolo_image = cv2.resize(yolo_image, dsize=shape)

		# scale pixel values to [0, 1]
		yolo_image = yolo_image.astype('float32')
		yolo_image /= 255.0
		# add a dimension so that we have one sample
		yolo_image = expand_dims(yolo_image, 0)
		return yolo_image, width, height

	# get all of the results above a threshold
	def get_boxes(self,boxes, labels, thresh):
		v_boxes, v_labels, v_scores = list(), list(), list()
		# enumerate all boxes
		for box in boxes:
			# enumerate all possible labels
			for i in range(len(labels)):
				# check if the threshold for this label is high enough
				if box.classes[i] > thresh:
					v_boxes.append(box)
					v_labels.append(labels[i])
					v_scores.append(box.classes[i]*100)
					# don't break, many labels may trigger for one box
		return v_boxes, v_labels, v_scores


	def yolo_process(self,image):
		input_w, input_h = 416, 416
		yolo_image, image_w, image_h = self.load_image_pixels(image, (input_w, input_h))

		# make prediction
		yhat = self.model.predict(yolo_image, verbose=0)
		# summarize the shape of the list of arrays
		#print([a.shape for a in yhat])
		# define the anchors
		anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
		# define the probability threshold for detected objects
		class_threshold = 0.6
		boxes = list()
		for i in range(len(yhat)):
			# decode the output of the network
			boxes += self.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

		# correct the sizes of the bounding boxes for the shape of the image
		self.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
		# suppress non-maximal boxes
		self.do_nms(boxes, 0.5)
		# define the labels

		# get the details of the detected objects
		v_boxes, v_labels, v_scores = self.get_boxes(boxes, self.labels, class_threshold)
		# summarize what we found
		return v_boxes, v_labels, v_scores

class StdYOLO(BaseYOLO):
	def __init__(self):
		self.model = load_model('std-yolo.h5')
		
		self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
			"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
			"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
			"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
			"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
			"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
			"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
			"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
			"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class FaceYOLO(BaseYOLO):
	def __init__(self):
		self.model = load_model('face-yolo.h5')
		self.labels = ["face"]
			