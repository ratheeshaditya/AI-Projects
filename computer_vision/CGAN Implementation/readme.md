Rough implementation of Conditional GANS

Providing the additional information Y which is the target label, to be able to generate controlled output based on the target label. Unlike GAN, where we have no control over what is being generated.

Ref paper : https://arxiv.org/abs/1411.1784

![alt text](https://blogs.rstudio.com/ai/posts/2018-09-20-eager-pix2pix/images/pix2pixlosses.png)




I've trained this on the Anime dataset(which can be found here :- https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)

![alt text](https://i.ibb.co/KzPJ1k3/45-3999.png)
![alt text](https://i.ibb.co/QQQp2DR/46-999.png)
![alt text](https://i.ibb.co/6D2fL9h/46-2999.png)

