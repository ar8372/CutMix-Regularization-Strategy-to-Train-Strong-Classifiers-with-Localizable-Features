<h1 align="center">
  <b>CutMix-Regularization</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.5-ff69b4.svg" /></a>
       <a href= "https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>

</p>
Through this project we will try to understand CutMix Paper by implementing it on a simple problem of cat-vs-dog classification. 

## Implementation
I have used keras to implement the cutmix augmentation. 
Dataset used: https://www.kaggle.com/competitions/dogs-vs-cats/data. <br>
In CutMix we mix two images and their labels respectively in order to generate new data point. So I have created a custom datagenerator. This custom datagenerator takes batches from two seperate keras datagenerator and returns one batch by applying cutmix on it.<br>
>train datagenerator (custom) 
```
train_dataset = CutMixImageDataGenerator(
    generator1= train_datagen1, #train_generator1,
    generator2= train_datagen2, #train_generator2,
    img_size=128,
    batch_size=32,
)
```
>valid datagenerator (default) 
```
valid_dataset = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=image_path,
    target_size=(128, 128), 
    x_col='filename', 
    y_col='category',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",  
    )
```    

## Results
Without CutMix             |  With CutMix
:-------------------------:|:-------------------------:
![](resources/0.png)  |  ![](resources/1.png)

CutMix augmentation Final output: 
<p>
    <img src="resources/1.png" />
</p>

## Reference
- original paper: https://arxiv.org/abs/1905.04899
