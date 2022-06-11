#import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os
import config

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))
		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		return predMask

def predict(model,image):
	model.eval()
	with torch.no_grad():
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image1 = image1.astype("float32") / 255.0
		image1 = np.transpose(image1, (2, 0, 1))
		image1 = np.expand_dims(image1, 0)
		image1 = torch.from_numpy(image1).to(config.DEVICE)
		predMask = model(image1.float()).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
	return predMask



def draw_contour(img,mask):
	img_copy=img.copy()
	bldng_contours=cv2.Canny(mask.astype(np.uint8),100,200)
	contours,h=cv2.findContours(bldng_contours,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(img_copy, contours, -1, (0,0,255), 1)
	return img_copy


def main():
	imagenames=os.listdir(images_path)	
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	for name in imagenames:
		img_path=os.path.join(images_path,name)
		msk_path=os.path.join(masks_path,name)
		annotated_img=os.path.join(savedir,name)
		img=cv2.imread(img_path)
		msk=cv2.imread(msk_path)
		img_mask=draw_contour(img,msk)
		cv2.imwrite(annotated_img,img_mask)

def test_images_predictions(model,image_paths_list):
	
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	for img_pth in image_paths_list:
		img=cv2.imread(img_pth)
		predMask=predict(model,img)
		annotated_img=draw_contour(img,predMask)
		cv2.imwrite(os.path.join(savedir,img_pth.split('/')[-1]),annotated_img)

if __name__=='__main__':
	images_path='/home/bmuhwezi/building_segmentation/GEP_training_data/train_img_tuning/'
	masks_path='/home/bmuhwezi/building_segmentation/GEP_training_data/train_masks_tuning/'
	savedir='annotated_images'
	#main()	
	img=cv2.imread(os.path.join(images_path,os.listdir(images_path)[0]))
	predMask=predict(unet,img)
	a=draw_contour(img,predMask)
