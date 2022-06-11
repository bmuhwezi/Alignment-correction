import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

root_directory = 'Data'

patch_size = 256

def create_image_patches():
	#Read images from repsective 'images' subdirectory
	#As all images are of ddifferent size we have 2 options, either resize or crop
	#But, some images are too large and some small. Resizing will change the size of real objects.
	#Therefore, we will crop them to a nearest size divisible by 256 and then 
	#divide all images into patches of 256x256x3. 
	image_dataset = []  
	I=0
	for path, subdirs, files in os.walk(root_directory):
		if I>2:
			break
		dirname = path.split(os.path.sep)[-1]
		if dirname == 'masks':   #Find all 'images' directories
			images = os.listdir(path)  #List of all image names in this subdirectory
			for i, image_name in enumerate(images):  
				if image_name.endswith(".png"):   #Only read jpg images...
			       
					image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
					#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
					SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
					SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
					image = Image.fromarray(image)
					image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
					#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation      
					image = np.array(image)               
					#Extract patches from each image
					print("Now patchifying image:", path+"/"+image_name)
					patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
					
					for i in range(patches_img.shape[0]):
						for j in range(patches_img.shape[1]):
						
							single_patch_img = patches_img[i,j,:,:]
							
							#Use minmaxscaler instead of just dividing by 255. 
							#single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
							
							#single_patch_img = (single_patch_img.astype('float32')) / 255. 
							single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
							cv2.imwrite('masks/'+image_name.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',single_patch_img)
							#image_dataset.append(single_patch_img)
			I=I+1	

create_image_patches()
