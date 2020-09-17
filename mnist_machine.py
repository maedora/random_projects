from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

#test_image = cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2GRAY)
#noise = np.linspace(0.1, 3, 20)

noise = [0.1, 0.5, 0.75, 0.9, 1.25, 3]

# y-axis of the linfoot plot
fidel = []
struc = []
corre = []

for noise_factor in noise:
#noise_factor = 0.1
    
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    #x_test_noisy =  test_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)


    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    """


    filepath="mnist-model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


    history = autoencoder.fit(x_train_noisy, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=callbacks_list)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """

    # change directory if you are not using Chonki
    weights = 'D:/ACADS/machine_learning/project3_poisson/mnist-model-50-0.0747.hdf5'
    autoencoder.load_weights(weights)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')
    recon_img = autoencoder.predict(x_test_noisy)

    """
    def rgb2gray(rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    """

    # Plotting the resulting images side by side
    fig=plt.figure(figsize=(8, 4))
    columns = 3
    rows = 1
    fig.add_subplot(rows, columns, 1)
    noisy = x_test_noisy[1]
    noisy = np.reshape(noisy, [28,28])
    #noisy = rgb2gray(noisy)
    #noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    plt.title("Noisy", fontsize=18)
    plt.imshow(noisy)
    plt.axis('off')

    fig.add_subplot(rows, columns, 2)
    #cv_clean = np.reshape(noisy, [28,28])
    #cv_clean = cv2.fastNlMeansDenoisingColored(noisy, None, 28, 28, 7, 15)
    orig = x_test[1]
    orig = np.reshape(orig, [28,28])
    plt.title("Original", fontsize=18)
    plt.imshow(orig)
    plt.axis('off')

    fig.add_subplot(rows, columns, 3)
    clean = recon_img[1]
    clean = np.reshape(clean, [28,28])
    #clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    plt.title("Reconstructed", fontsize=18)
    plt.imshow(clean)
    plt.axis('off')

    #height = clean.shape[0]
    #width = clean.shape[1]
    #channels = clean.shape[2]
    #print(height, width, channels)
    plt.savefig('noise' + str(noise_factor) + '.png', bbox_inches='tight')
 
"""


    ######## Getting Linfoot's criteria of the image ##########
    orig = x_test[1]
    orig = np.reshape(orig, [28,28])
    #orig = np.mean(orig)

    clean = recon_img[1]
    clean = np.reshape(clean, [28,28])
    #clean = np.mean(clean)

    # Fidelity measures the degree of similarity between the true value and estimated value
    def fidelity(orig, cln):
        F = 1 - (np.mean(np.sqrt((orig-cln)**2))/np.mean(np.sqrt(orig**2)))
        return F
    # Structural content measures the sharpness between two intensities
    def structure(orig, cln):
        T = np.mean(np.sqrt(cln**2))/np.mean(np.sqrt(orig**2))
        return T
    # Correlation quality gives a measure of the degree of alignment of the peaks
    def correlate(orig, cln):
        Q = np.mean(np.sqrt((orig*cln)**2))/np.mean(np.sqrt(orig**2))
        return Q

    fidel.append(fidelity(orig,clean))
    struc.append(structure(orig,clean))
    corre.append(correlate(orig,clean))

plt.plot(noise, fidel, marker = "s", color = "k", label = "Fidelity")
plt.plot(noise, struc, marker = "o", color = "k", label = "Structural content")
plt.plot(noise, corre, marker = "v", color = "k", label = "Correlation quality")
plt.xlabel("Noise level")
plt.legend()

plt.show()

    
# print(fidelity(orig, clean))
# print(structure(orig, clean))
# print(correlate(orig, clean))

"""

"""
# Getting the psnr of the images 
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(orig, clean)
print(d)
"""
