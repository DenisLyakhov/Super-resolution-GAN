from model import SRModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, convert_image_dtype, resize, random_crop

from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.optimizers import Adam

import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset_input', type=str)

imageSize = None

def pretrainModel(model, lowRes, highRes):
    with tf.GradientTape() as tape:
        upscaledRes = model.generatorModel(lowRes)
        meanSquaredError = MeanSquaredError()(highRes, upscaledRes)

    gradient = tape.gradient(meanSquaredError, model.getGeneratorVariables())
    model.generatorOptimizer.apply_gradients(zip(gradient, model.getGeneratorVariables()))

def normalizeImage(img):
    #Scale image from [-1, 1] to [0, 255]
    return (img + 1.) / 2. * 255

def calculateContentLoss(model, highRes, upscaledRes):
    rescaleFactor = 12.75

    upscaledRes = keras.applications.vgg19.preprocess_input(normalizeImage(upscaledRes))
    highRes = keras.applications.vgg19.preprocess_input(normalizeImage(highRes))

    mse = tf.keras.losses.MeanSquaredError()(model.vgg(highRes) / rescaleFactor, model.vgg(upscaledRes) / rescaleFactor)

    return mse

def trainOnSample(model, lowRes, highRes):
    size = model.img_size // 16

    highResRef = tf.ones((lowRes.shape[0],) + (size, size, 1))
    upscaledResRef = tf.zeros((lowRes.shape[0],) + (size, size, 1))

    with tf.GradientTape() as generatorTape, tf.GradientTape() as discriminatorTape:
        upscaledRes = model.generatorModel(lowRes)

        highResPred = model.discriminatorModel(highRes)
        upscaledResPred = model.discriminatorModel(upscaledRes)

        originalResLoss = BinaryCrossentropy()(highResRef, highResPred)
        upscaledResLoss = BinaryCrossentropy()(upscaledResRef, upscaledResPred)
        meanLoss = tf.add(originalResLoss, upscaledResLoss)

        perceptualLoss = tf.keras.losses.MeanSquaredError()(highRes, upscaledRes)

        # Content loss + Adversarial loss
        perceptualLoss += calculateContentLoss(model, highRes, upscaledRes) + BinaryCrossentropy()(highResRef, upscaledResPred) * 0.001

    # Perceptual loss
    generatorGradient = generatorTape.gradient(perceptualLoss, model.getGeneratorVariables())
    discriminatorGradient = discriminatorTape.gradient(meanLoss, model.getDiscriminatorVariables())

    model.generatorOptimizer.apply_gradients(zip(generatorGradient, model.getGeneratorVariables()))
    model.discriminatorOptimizer.apply_gradients(zip(discriminatorGradient, model.getDiscriminatorVariables()))


def trainEpoch(model, dataset):
    for lowRes, highRes in dataset:
        trainOnSample(model, lowRes, highRes)

        if model.iter % 200 == 0:
            # TODO: print(losses)
            print('step')
        model.iter += 1


def parseAndPreprocess(path):
    downsamplingFactor = 4
    rescaledSize = imageSize // downsamplingFactor

    # Read images
    highRes = read_file(path)
    highRes = decode_jpeg(highRes, channels=3)
    highRes = convert_image_dtype(highRes, tf.float32)

    cond = tf.reduce_all(tf.shape(highRes)[:2] >= tf.constant(imageSize))

    highRes = tf.cond(cond, lambda: tf.identity(highRes), lambda: resize(highRes, [imageSize, imageSize]))

    #Preprocessing
    highRes = random_crop(highRes, [imageSize, imageSize, 3])
    lowRes = resize(highRes, [rescaledSize, rescaledSize], method='bicubic')

    # Scale image from [0, 1] to [-1, 1]
    highRes = highRes * 2. - 1.

    return lowRes, highRes


def loadDataset(path):
    image_paths = []

    for img in os.listdir(path):
        image_paths.append(os.path.join(path, img))

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(parseAndPreprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds.shuffle(30).batch(2, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

def initializeModelOptimizers(model):
    generatorSchedule = keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=100000, decay_rate=0.1, staircase=True)
    model.generatorOptimizer = Adam(learning_rate=generatorSchedule)

    discriminatorSchedule = keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=100000, decay_rate=0.1, staircase=True)
    model.discriminatorOptimizer = Adam(learning_rate=discriminatorSchedule)

def trainModelOnDataset(model, dataset, epochs=10):
    # Pretrain
    for lowRes, highRes in dataset:
        pretrainModel(model, lowRes, highRes)

    for i in range(1, epochs+1):
        print('EPOCH: ' + str(i))
        trainEpoch(model, dataset)
        model.saveModel('models/')


# Usage: python train.py --dataset_input './input/'
def main():
    global imageSize

    args = parser.parse_args()

    imageSize = 384
    epochs = 20
    #trainDir = '../../dataset/train_high'

    ds = loadDataset(args.dataset_input)

    srModel = SRModel()
    
    srModel.initModel(imageSize)
    initializeModelOptimizers(srModel)

    trainModelOnDataset(srModel, ds, epochs)


if __name__ == '__main__':
    main()
