import torch
from UNet import Generator
from discriminator import Discriminator
from dataset import AnimeDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.optim import Adam,SGD
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Resize,Normalize,Compose



if __name__ == "__main__":
    DATASET_PATH = r"D:\My Projects\ML\Dataset\Anime\data\train"
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    #Defining the output channles of the generator
    
    channels = (3,64,128,256,512,512,512,512)
    
    gen_unet = Generator(
        channels
    ).to(device)
    
    #Defining the discriminator
    disc = Discriminator().to(device)

   
    #Defining hyperparameters of the model
    num_epochs = 100
    learning_rate =  0.0002
    batch_size = 1
    lambda_ = 100 #The lambda parameter for the generator

    #defining loss and optimizer
    l1 = L1Loss() #For the pixel wise loss for the generator
    bceloss = BCEWithLogitsLoss() 

    optimizer_g = Adam(gen_unet.parameters(),lr = learning_rate,betas=(0.5, 0.999))
    optimizer_d = Adam(disc.parameters(), lr = learning_rate,betas=(0.5, 0.999))
    
    #Dataset definition & its corresponding transformations
    transform = Compose([
        ToTensor(),
        Resize((256,256)),
        Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset_whole = AnimeDataset(DATASET_PATH,transforms=transform,x="edge")
    dataset = torch.utils.data.Subset(dataset_whole, torch.randint(0,len(dataset_whole),(5000,)))

    loader = DataLoader(dataset, batch_size=batch_size)

    #defining the training loop
    
    for epoch in range(num_epochs):
        print(f"[+]Running for epochs : {epoch+1}/{num_epochs}")
        print(f"Running across : {len(loader)} samples.")    
        for iter, (img,label) in enumerate(loader):
            # define each iteration
            img = img.to(device)
            label = label.to(device)


            #Defining the discriminator
            y_fake = gen_unet(img)
            disc_real = disc(img,label)
            disc_fake = disc(img,y_fake.detach())
          
            disc_real_loss = bceloss(disc_real,torch.ones_like(disc_real)) #D(x)
            disc_fake_loss = bceloss(disc_fake,torch.zeros_like(disc_fake)) #(1-D(g(x)))
            discriminator_loss = disc_real_loss + disc_fake_loss
 
            disc.zero_grad()
            discriminator_loss.backward()
            optimizer_d.step()
            # break

            #Defining the generator
            
            gen_output = disc(img,y_fake)
            gen_fake_loss = bceloss(gen_output,torch.ones_like(gen_output))
            l1_loss = lambda_ *  l1(y_fake,label)
            generator_loss =  gen_fake_loss + l1_loss
            gen_unet.zero_grad()
            generator_loss.backward()
            optimizer_g.step()
   
            if (iter+1)%500 == 0:
                print(f"Discriminator loss : {discriminator_loss.item()} , Generator loss : {generator_loss.item()}")
                grid = make_grid(torch.cat([img,label,y_fake]), nrow=10).permute(1,2,0).cpu().numpy()
                plt.imshow(grid,cmap="gray")
                plt.savefig(f"sample_img/{epoch}-{iter}.png")
  
