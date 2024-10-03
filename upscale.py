from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import cv2
import os
from glob import glob

import time
import random

parser = ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

# Usage: python generate.py --input './input/' --output './output/'
def main():
    args = parser.parse_args()

    fileNames = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    model = keras.models.load_model('models/generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    for fName in fileNames:
        
        input = cv2.imread(fName, 1)

        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = input / 255.0

        pred = model.predict(np.expand_dims(input, axis=0))[0]

        pred = (((pred + 1) / 2) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(fName)), pred)
        


if __name__ == '__main__':
    main()
