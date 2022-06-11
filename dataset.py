# import the necessary packages
import numpy as np
import cv2
from torch.utils.data import Dataset
import os



class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms=None,alignment=False):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths=imagePaths
		self.maskPaths=maskPaths
		#self.imageids = os.listdir(imagePath)
		self.transforms = transforms
		self.alignment =alignment
	
	def __len__(self):
			
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	
	def __getitem__(self, idx):
			
		# grab the image path from the current index
		#imagePath = os.path.join(self.imagePaths,self.imageids[idx])
		#maskPath = os.path.join(self.maskPaths,self.imageids[idx])
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(maskPath,0)
		# Check to see if there is alignment correction and add an alpha channel to the image	
		if self.alignment:
			x_shift=np.random.randint(-10,10)
			y_shift=np.random.randint(-10,10)
			mask_translation=self.translate_mask(mask,x_shift,y_shift)
			image=np.dstack((image,mask_translation))

		# check to see if we are applying any transformations
		if self.transforms is not None:
		# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

		# return a tuple of the image and its mask
		return (image,mask)

	def translate_mask(self, msk,x_shift,y_shift):

		T=np.float32([[1,0,x_shift],[0,1,y_shift]])
		translation=cv2.warpAffine(msk, T, msk.shape)
		return translation
	    

def extract_footprints(msk,buffer_size=10):
	
	ret,thresh = cv2.threshold(msk,1,255,cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)
	lst=[]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		lst.append([x-buffer_size,y-buffer_size,x+w+buffer_size,y+h+buffer_size])	
	return lst
	
def main():
	msks=os.listdir(msk_path)
	save_msk_dir=os.path.join(save_dir,'msk_patches')
	save_img_dir=os.path.join(save_dir,'img_patches')
	if not os.path.exists(save_msk_dir):
		os.mkdir(save_msk_dir)
	if not os.path.exists(save_img_dir):
		os.mkdir(save_img_dir)
	for msk_name in msks:
		msk=cv2.imread(os.path.join(msk_path,msk_name),0)
		img=cv2.imread(os.path.join(img_path,msk_name))
		bldng_patchs=extract_footprints(msk,20)
		for i in bldng_patchs:
			x1,y1,x2,y2=i
			x1=np.clip(x1,0,len(msk)-1)
			y1=np.clip(y1,0,len(msk)-1)
			x2=np.clip(x2,0,len(msk)-1)
			y2=np.clip(y2,0,len(msk)-1)
			msk1=msk[y1:y2,x1:x2]		
			img1=img[y1:y2,x1:x2]	
			sffix=str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)+'_'+str(msk1.shape[0])+'-by-'+str(msk1.shape[1])+'_'	
			cv2.imwrite(os.path.join(save_msk_dir,sffix+msk_name),msk1)	
			cv2.imwrite(os.path.join(save_img_dir,sffix+msk_name),img1)	

if __name__=='__main__':
	
	msk_path='/home/bmuhwezi/building_segmentation/GEP_training_data/train_masks_tuning'
	img_path='/home/bmuhwezi/building_segmentation/GEP_training_data/train_img_tuning'
	save_dir='/home/bmuhwezi/building_segmentation/GEP_training_data/'




