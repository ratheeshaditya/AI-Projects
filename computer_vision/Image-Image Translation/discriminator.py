from torch import nn
import torch


class CNNBlock(nn.Module):
    """
        @CNNBlock
        Parameters:-
        ---------------------
        in_channels:- Input size(int)
        out_channels :- Output size of the convolutions(int)
    """
    def __init__(self,in_channels,out_channels,stride=2):
        super(CNNBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,bias=False,padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.block(x)



class Discriminator(nn.Module):
    """
        Implementation of the discriminator for image-image translation tasks, which also adopts the UNet architecture
        @Discriminator
        Parameters
        ---------------
        channels:- Input channels, i.e, number of image channel(3 or 1)
        feature_list:- Takes in a list of features for the convolution blocks
    """

    def __init__(self,channels=3,feature_list = [64,128,256,512]):
        super(Discriminator,self).__init__()
        #we are doing channel * 2 to take into account for the "y", label, sicne we are essentially concatenation
        #them before passing them as input to the discriminator.
        self.initial= nn.Sequential(
            nn.Conv2d(channels * 2, feature_list[0],kernel_size=4,stride=2,
            padding=1,padding_mode="reflect"),
             nn.LeakyReLU(0.2)
        )
        layers = []
        in_channel = feature_list[0]
        for feature in feature_list[1:]:
            # we do a stride of 1 if we reach the final layer, otherwise we use a stride of 2
            layers.append(CNNBlock(in_channel,feature, stride=1 if feature==feature_list[-1] else 2)) 
            in_channel = feature
        #appending the final output, to make sure it output of 1 class is predicted
        layers.append(nn.Conv2d(feature_list[-1],1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
        

    def forward(self,x,y):
        #we have the image x and y we concatenate them now
        input_ = torch.concat([x,y],dim=1) #concatenate along the channel axis
        output = self.initial(input_)
        output = self.model(output)
        return output




if __name__ == "__main__":
    x = torch.randn(1,3,256,256)
    y = torch.randn(1,3,256,256)
    model = Discriminator()
    print(model(x,y).shape)

