<h3>Image-Image translation</h3>
Implementation of the Image-Image translation paper in PyTorch :- https://arxiv.org/abs/1611.07004

In this paper, they address the problem of image-image translation by using conditional gans. They do this by conditioning the discriminator with the target image, and also introducing an additional loss term(L1 loss) to penalize the generator to generate finely-grained images. 



![alt text](https://blogs.rstudio.com/ai/posts/2018-09-20-eager-pix2pix/images/pix2pixlosses.png)



<h3>Predicted Images</h3>
I've trained this on the Anime dataset(which can be found here :- https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair).

Below are some examples of the predictions that the model made. The first column in the image is the input, second column is the target image, and the last column is the generated image. Note the model has been trained only for around ~50 epochs to get this result. 

![alt text](https://i.ibb.co/KzPJ1k3/45-3999.png)
![alt text](https://i.ibb.co/QQQp2DR/46-999.png)
![alt text](https://i.ibb.co/6D2fL9h/46-2999.png)
