# Face-Frontalization

Project theme:

    An AI-based solution to generate front profile face from side profile face.

Intern:    

    Sri Rama Chandra Reddy Baddigam 
[contact](mailto:ep19btech11003@iith.ac.in) 

Goal: 

    To create a model that generates front profile images from side profile images with large pose angles.

Technologies:

•	Tensorflow-Keras

•	Mediapipe 

•	Dlib

•	Opencv 


Proposed solution:

 After going through multiple research papers, it is decided to implement a generative adversarial network. Going into the specifics, the model is chosen from the paper[], and the model is TP_GAN. It is a two-pathway generative adversarial network where the generator consists of local and global pathways. The local pathway is to accommodate changes in local features such as nose, eyes, and mouth while altering the pose-angle. The global pathway accommodates global changes such as facial structure.


Architecture:

 The idea of a GAN is a zero-sum game. The discriminator tries to differentiate authentic images from generated images. The discriminator's better performance increases the generator's loss and vice-versa. The generator takes the input: a side profile image and the four extracted features: left eye, right eye, nose, and mouth. The side profile image is passed through the global pathway consisting of a set of convolutional layers and then a group of deconvolutional layers. The four extracted features are passed through a local pathway each; these outputs are combined and concatenated with the output of deconvolution layers from the generator. Then a couple of convolution layers are applied.



Environment requirements[requirements.txt]:

•	opencv-python~=4.6.0.66

•	tensorflow~=2.9.1

•	numpy~=1.22.4

•	keras~=2.9.0

•	Pillow~=9.2.0

•	matplotlib~=3.5.2

•	imageio~=2.19.3

•	ipython~=8.4.0

•	imutils~=0.5.4

•	dlib~=19.24.0

•	mediapipe~=0.8.10

Pipeline:

 The function main.py takes arguments path to the saved model, either 'path_to_image' or 'path_to dir': to indicate whether the path belongs to image or object and followed by the path to image or directory, respectively. The generate function reads the image from the path provided and estimates the angle of the pose in the image. The model takes only images facing left; the right-facing images are flipped. The face is extracted from the image. Now feature extraction is performed on the extracted face. The input image and the features are passed to the model. The generated image is returned. 


Sample code: python main.py  ‘<model_path>’  path_to_image  ‘<image_path>’

###############################

for feature extraction the landmarks file can be found at [drive folder](https://drive.google.com/drive/folders/14C1xOl4DmN3cA2Xu8bBB-ad-jV_mMXx9?usp=sharing) , file named shape_predictor_68_face_landmarks.dat
