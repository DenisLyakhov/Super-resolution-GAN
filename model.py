from tensorflow import keras
from tensorflow.keras.layers import Conv2D, LeakyReLU, PReLU, BatchNormalization, Activation, UpSampling2D, Add, Input


class SRModel:
    def initModel(self, img_size):
        self.img_size = img_size
        self.upscaleFactor = 4
        self.iter = 0

        self.generatorOptimizer = None
        self.discriminatorOptimizer = None

        vggConvLayerName = 'block5_conv4'

        vggModel = keras.applications.VGG19(weights='imagenet', input_shape=(self.img_size, self.img_size, 3), include_top=False)
        self.vgg = keras.models.Model(inputs=vggModel.input, outputs=vggModel.get_layer(vggConvLayerName).output)

        self.discriminatorModel = self.createDiscriminatorModel()
        self.generatorModel = self.createGeneratorModel()

    def createDiscriminatorModel(self):
        a = 0.2
        m = 0.8
        shape = (self.img_size, self.img_size, 3)

        X_input = Input(shape=shape)

        x = Conv2D(64, kernel_size=(3,3), padding='same')(X_input)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(64, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(128, kernel_size=(3,3), strides=1, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(128, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(256, kernel_size=(3,3), strides=1, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(256, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(512, kernel_size=(3,3), strides=1, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        x = Conv2D(512, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=m)(x)
        x = LeakyReLU(alpha=a)(x)

        output = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(x)

        return keras.models.Model(X_input, output)
    
    def residualBlock(self, X_input):
        c = X_input

        c = Conv2D(64, kernel_size=(3,3), padding='same')(c)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)

        c = Conv2D(64, kernel_size=(3,3), padding='same')(c)
        c = BatchNormalization()(c)

        return Add()([X_input, c])

    def createGeneratorModel(self):
        shape = (self.img_size // self.upscaleFactor, self.img_size // self.upscaleFactor, 3)

        X_input = keras.Input(shape=shape)

        x1 = Conv2D(64, kernel_size=(9,9), padding='same')(X_input)
        x1 = BatchNormalization()(x1)
        x1 = PReLU(shared_axes=[1, 2])(x1)

        s = self.residualBlock(x1)
        s = self.residualBlock(s)
        s = self.residualBlock(s)
        s = self.residualBlock(s)
        s = self.residualBlock(s)
        s = self.residualBlock(s)
        s = self.residualBlock(s)

        x2 = Conv2D(64, kernel_size=(3,3), padding='same')(s)
        x2 = BatchNormalization()(x2)
        x2 = Add()([x2, x1])
        
        #bilinear
        x2 = UpSampling2D(interpolation='bilinear')(x2)
        x2 = Conv2D(64, kernel_size=(3,3), padding='same')(x2)
        x2 = PReLU(shared_axes=[1, 2])(x2)

        x2 = UpSampling2D(interpolation='bilinear')(x2)
        x2 = Conv2D(64, kernel_size=(3,3), padding='same')(x2)
        x2 = PReLU(shared_axes=[1, 2])(x2)

        output = Conv2D(3, kernel_size=(9,9), padding='same', activation='tanh')(x2)

        return keras.models.Model(X_input, output)
    
    def getGeneratorVariables(self):
        return self.generatorModel.trainable_variables
    
    def getDiscriminatorVariables(self):
        return self.discriminatorModel.trainable_variables
    
    def saveModel(self, path):
        self.generatorModel.save(path + 'generator.h5')
        self.discriminatorModel.save(path + 'discriminator.h5')
