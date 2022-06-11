from dataset import SegmentationDataset
import config
from model import UNet
#from Unet2 import Unet as UNet
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
#import matplotlib.pyplot as plt
from tqdm import tqdm
from imutils import paths
import torch
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional import jaccard_index


def iou_pytorch(outputs, labels):
	# You can comment out this line if you are passing tensors of equal shape
	# But if you are passing output from UNet or something it will most probably
	# be with the BATCH x 1 x H x W shape
	SMOOTH=0.0001
	#outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
	#intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
	intersection = torch.logical_and(labels, outputs)
	union = torch.logical_or(labels, outputs)
	#union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
	iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
	thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
	return thresholded  # Or thresholded.mean() if you are interested in average across the batch
	#return iou  # Or thresholded.mean() if you are interested in average across the batch
    
#load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
print('[INFO] image path: '+os.path.dirname(imagePaths[0]))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
print('[INFO] mask  path: '+os.path.dirname(maskPaths[0]))


intersects=list(set([i.split('/')[-1] for i in maskPaths]).intersection([i.split('/')[-1] for i in imagePaths]))
imagePaths=sorted([os.path.join(config.IMAGE_DATASET_PATH,i) for i in intersects])
maskPaths=sorted([os.path.join(config.MASK_DATASET_PATH,i) for i in intersects])

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
#f = open(config.TEST_PATHS, "w")
#f.write("\n".join(testImages))
#F.close()
# define transformations
transforms1 = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),transforms.RandomHorizontalFlip(p=0.9),transforms.RandomRotation(40),transforms.RandomVerticalFlip(p=0.9)])
transforms2 = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms2)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms2)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
# initialize our UNet model
unet = UNet().to(config.DEVICE)
#unet.apply(initialize_weights)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = -(-len(testDS) // config.BATCH_SIZE)
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": [],'accuracy':[]}
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	totalacc=0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		#loss = BCEWithLogitsLoss(pos_weight=(y==0.).sum()/y.sum())(pred,y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
			pred_loss=torch.sigmoid(pred)>config.THRESHOLD
			#totalacc += (255*(torch.sigmoid(pred)>0.2)).data.eq(y.data).sum().item()/y.nelement()
			#totalacc += ((2*(pred_loss*y.data)).sum())/((y+pred_loss).sum()+0.000001)
			totalacc +=jaccard_index((pred_loss*1),y.int(), num_classes=2)	
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	Avgacc=totalacc/testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	H['accuracy'].append(Avgacc)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))



####### Plot the loss 
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on LR: "+str(config.INIT_LR))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH);plt.close()
plt.plot(H['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Mean IOU')
plt.savefig('plot_accuracy.png');plt.close()
# serialize the model to disk
torch.save(unet, config.MODEL_PATH)


