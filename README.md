# Super Resolution Image Upscaler

The goal of this project was to implement a real time image upscaler model based on SRGAN architecture ([link](https://arxiv.org/pdf/1609.04802)).

## Model predictions

![Low_res_pic (5) (1)](https://github.com/user-attachments/assets/039cd5e7-0ee4-4b9b-ae39-3e850f3b8d53)
![Upscaled_pic (5)](https://github.com/user-attachments/assets/62ad56db-4341-425f-beec-73f2a0dbd221)

![Low_res_pic_1](https://github.com/user-attachments/assets/47157785-76d4-4ef7-b52b-c9887b2546bd)
![Upscaled_pic_1](https://github.com/user-attachments/assets/eecd0c5f-0e53-4964-b7a5-2502861c6da9)

## Requirements

```
pip install tensorflow==2.10.0 opencv-python==4.1.1.26 numpy==1.17.2
```

## Training

```
python train.py --dataset_input './input_directory/'
```

## Upscaling images

```
python upscale.py --input './input_directory/' --output './outpu_directory/'
```

Pre-trained model should be placed in the "models" folder.
