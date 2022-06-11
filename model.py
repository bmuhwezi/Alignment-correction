import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch import nn





class Block(Module):
	
	def __init__(self,inchannels,outchannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1=Conv2d(inchannels,outchannels,3,padding='same')
		self.relu=ReLU()
		#self.Batch_Norm=nn.BatchNorm2d(outchannels,momentum=0.01,affine=False)
		self.conv2=Conv2d(outchannels,outchannels,3,padding='same')

	def forward(self,x):
		# apply CONV => RELU => CONV block to the inputs and return it
		#return self.conv2(self.Batch_Norm(self.relu((self.conv1(x)))))
		return self.conv2(self.relu((self.conv1(x))))


class Encoder(Module):
	
	def __init__(self,channels=(3, 16, 32, 64)):
		super().__init__()
		self.encBlocks=ModuleList([Block(channels[i],channels[i+1]) for i in range(len(channels)-1)])	
		self.pool=MaxPool2d(2,stride=2)

	def forward(self,x):
		blockOutputs=[]
		for block in self.encBlocks:
			x=block(x)
			blockOutputs.append(x)
			x=self.pool(x)
		return blockOutputs


class Decoder(Module):

	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		#lst=[]
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
			#lst.append(x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures


class UNet(Module):

	def __init__(self, encChannels=(3,16,32,64,128,256),
		 decChannels=(256,128,64,32,16),
		 nbClasses=1, retainDim=True,
		 outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		self.initialize_weights()

	def forward(self, x):
			# grab the features from the encoder
			encFeatures = self.encoder(x)
			# pass the encoder features through decoder making sure that
			# their dimensions are suited for concatenation
			decFeatures = self.decoder(encFeatures[::-1][0],
				encFeatures[::-1][1:])
			# pass the decoder features through the regression head to
			# obtain the segmentation mask
			map = self.head(decFeatures)
			# check to see if we are retaining the original output
			# dimensions and if so, then resize the output to match them
			if self.retainDim:
				map = F.interpolate(map, self.outSize)
			# return the segmentation map
			return map

	def initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias.data, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight)
				nn.init.constant_(m.bias.data, 0)
