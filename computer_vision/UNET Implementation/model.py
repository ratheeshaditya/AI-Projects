import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of UNet
@Author - Aditya R
"""



class DoubleConv(nn.Module):
    """

    @DoubleConv -> Class that takes input, performs double convolution followed by ReLU layer

    """
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3),
            nn.ReLU()
        )
    def forward(self,x):
        
        return self.block(x)
    
class Encoder(nn.Module):
    """
        @Encoder -> Class that creates an instance of encoder based on the number of channels.
        args :- out_channel - tuple of integer
        Example:- out_channel = (1,64,128,256,512,1024)
        - The forward function takes in input, applies double convolution and performs max-pooling
        - Returns the appended features of the encoder for each channel
        

    """
    def __init__(self,out_channel):
        super().__init__()
        self.channels = out_channel
        self.encoderblock = nn.ModuleList([DoubleConv(self.channels[i],self.channels[i+1])
             for i in range(len(self.channels)-1) 
        ])
    
        self.maxpool = nn.MaxPool2d(2)
       
    def forward(self,x):
        features = [] #contains features in this ascending order. 1,64,128,256,512,1024
        for i,v in enumerate(self.encoderblock):
            if i!=len(self.channels):
                x = v(x)
                features.append(x)
                x = self.maxpool(x)
            else:
                x = v(x)
        return features

class Decoder(nn.Module):
    """
        @Decoder -> Class that creates an instance of encoder based on the number of channels.
        args :- out_channel - tuple of integer
        Example:- out_channel = (1024,512,256,64). The last channel gets removed because we replace it with the number of classes

        - The forward function takes in input, applies double convolution and performs max-pooling
        - Returns the appended features of the encoder for each channel
        

    """
    def __init__(self,out_channel):
        super().__init__()
        self.channels = out_channel[::-1][:-1] #reversing the channel tuples(1024,512,256,128,64)
        self.up_conv = nn.ModuleList([])
        self.dec_conv = nn.ModuleList([])
        #Initialize decoder block with up_conv and convolution blocks
        for i in range(len(self.channels)-1): 
            self.up_conv.append(nn.ConvTranspose2d(self.channels[i],self.channels[i+1],2,2))
            self.dec_conv.append(DoubleConv(self.channels[i],self.channels[i+1]))
        
    def _center_crop(self, feature: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
        """Crops the input tensor to the target size.
        """
        _, _, H, W = target_size.shape
        _, _, h, w = feature.shape

        # Calculate the starting indices for the crop
        h_start = (h - H) // 2
        w_start = (w - W) // 2

        # Crop and returns the tensor
        return feature[:, :, h_start:h_start+H, w_start:w_start+W]
    


    def forward(self,x,encfeatures): #x-> 1024, encfeatures -> (512,256,128,64)
        #iterating through the up_conv blocks and conv blocks and applying those functions on x
        for index,(upconv,conv) in enumerate(zip(self.up_conv,self.dec_conv)):
            x = upconv(x) #1024 -> 512(1st iteration) -> 256
        
            encoder_feature = self._center_crop(encfeatures[index], x) #512,h,w -> 256
            x = torch.cat([x, encoder_feature], dim=1) #1024 -> 512
            x = conv(x) #apply double conv to get 512 -> to something else
            # print(x.shape)
        return x


class UNet(nn.Module):
    def __init__(self,outchannels,no_class,out_size=(224,224),preserve=True):
        super().__init__()
        self.encoder_blocks = Encoder(outchannels)
        self.decoder_block = Decoder(outchannels)
        self.final_output = nn.Conv2d(outchannels[1], no_class, kernel_size=1)

        self.preserve = preserve
        self.outSize = out_size
        
    def forward(self,x):
        encfeatures = self.encoder_blocks(x)[::-1] #1024,512 , 256

        decoder = self.decoder_block(encfeatures[0],encfeatures[1:])
        output = self.final_output(decoder)
        if self.preserve:
            output = F.interpolate(output, self.outSize)
        return output



if __name__=="__main__":
    channels = (1,64,128,256,512,1024)
    decoder_block = UNet(channels,3).to("cuda")
    # decoder_block
    a = torch.randn(1,1,572,572).to("cuda")
    print(decoder_block(a).shape)
