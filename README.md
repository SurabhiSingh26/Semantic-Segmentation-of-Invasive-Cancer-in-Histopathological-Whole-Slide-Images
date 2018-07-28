# Semantic-Segmentation-of-Invasive-Cancer-in-Histopathological-Whole-Slide-Images
Segments different type of cancerous regions of an image of a tissue using semantic segmentation.

Implementation

	We implemented this model using tensorflow and openslide library in python. For reading images we used scipy and opencv. Our code uses input as .svs files for features and .xml file with coordinates of cancerous regions as label.
	
	Preprocessing needs to be done to convert .svs and .xml files into patches of [2000,2000,3] which can be fed into the model's place holders.We implemented preprocessing step in read_svs.py and read_xml.py files which divides the data into .jpg files of size [2000,2000,3].
	
	Batch Generation is done by gen_batch_function in WSI_Detection.py file. It return an iterator which generates features and labels of [batch size,2000,2000,3].
	
	Model is generated model in WSI_Detection.py file according to above architecture which returns segmented image as output.
	
	Optimizer: We used Adam optimizer and cross entropy loss in our model,both were implemented in optimizer in WSI_Detection.py
	
	Accuracy: We used IOU score as accuracy for our model. Tensorflow inbuilt functions are not compatible with our model so we designed our own IOU score method with finds intersection and union and calculates result. It was implemented in  find_IOU_score in WSI_Detection.py
	
	Separate training and testing files were created with relevant global variables. Trained model was stored in beanbag/model which can be directly imported for testing.
	
	For Testing: WSI_testing.py can be executed specifying proper directory names in the global variables of that file.	
	Requirements
	For testing make sure that PC fulfills following requirements:
	
			Python 3.6
			Scipy
			Openslide
			OpenCV
			Tensorflow 1.8
			Pip or Conda 
			Numpy
			Matplotlib
			Tensorboard
		
	Note: Make sure everything is updated.

	
  Before running Store .svs files in work folder and create  a dataset and beanbag folder.
