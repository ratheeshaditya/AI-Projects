import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of the generator side of image-image translation

"""
class TransposeBlock(nn.Module):
    """

    @TransposeBlock -> Class that takes input and transposes it given fixed kernel size, stride
    and applies a ReLU activation function.

    Parameters:
    ---------------
    in_channel : Input size (int)
    out_channel : Output size(int) 
    """
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel,out_channel,4,2,1),
            nn.ReLU()
            # nn.Dropout(0.2),

        )
    def forward(self,x,features):

        x = torch.concat([x,features],dim=1)
        return self.block(x)



class ConvolutionBlock(nn.Module):
    """

    @ConvolutionBlock :- Outputs the convolution of the input
    Parameters:-
    ---------------
    in_channel:- Input size(int)
    out_channel:- Output size(int)
    """
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,2,1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

        )
    def forward(self,x):
        return self.block(x)



class Generator(nn.Module):
    """
        @Generator :- Class that glues the pieces together
        Parameters
        ---------------
        out_channel :- Is a tuple of channels in increasing order (3,64,128,256,512)
    """
    def __init__(self,out_channel):
        super().__init__()
        self.channels = out_channel #Tuples of output channels
        # print(self.channels)
        self.reverse_channel = self.channels[::-1] #All the reversed channels
       

        self.encoderblock = nn.ModuleList([ConvolutionBlock(self.channels[i],self.channels[i+1])
             for i in range(len(self.channels)-1) 
        ])

        self.transposeblock = nn.ModuleList([TransposeBlock(self.reverse_channel[i]*2,self.reverse_channel[i+1])
             for i in range(len(self.reverse_channel)-1) 
        
        ]) #we multiply the inpuit channel *2 because we are concatenating

    
        self.bottleneck = nn.Conv2d(self.channels[-1],self.channels[-1],4,2,1) #makes the final output into the shape of 1x1
        self.initial_transpose = nn.Sequential(
            nn.ConvTranspose2d(self.channels[-1],self.channels[-1],4,2,1),
            nn.ReLU(),
            nn.Dropout(0.5),

        ) #we do this to upsample the output of the bottleneck to 2x2, i.e, to match the dimension of the final output of the downsampling.
        self.tanh = nn.Tanh()
       
    def forward(self,x):
        features = [] #contains features in this ascending order. 1,64,128,256,512,1024
        shapes = []
        for i,v in enumerate(self.encoderblock):
            x = v(x)
            features.append(x) #Appending all the features in order to apply skip-connections in later stages
         

        final = self.bottleneck(x) #The final bottle neck that outputs dimennsion of 1x1 
        final = self.initial_transpose(final) #match the dimensions to the previous layer(2x2), so do a transpose to get 1x1 to 2x2

        for feature,upsample in zip(features[::-1],self.transposeblock):
            final = upsample(final,feature)
            
        final = self.tanh(final)
        #now lets upsample this
        return final


if __name__=="__main__":
    channels = (3,64,128,256,512,512,512,512)
    encoder = Generator(channels)
    a = torch.randn(1,3,256,256)    

    c,d = encoder(a)
    # print(d[::-1])