import keras
from keras.utils import np_utils
from keras.preprocessing import image
import scipy
import os
import random
import numpy
from nn_lib import CNNLIB

class WordImages:
    DATA_PATH = "/home/drogaar/pyScripts/text2image/words_large"
    TESTSET_PATH = "../data/testset_words.csv"
    random.seed("convolution")
    numpy.random.seed(42)
    DATA_SIZE = 5000
    rgb_normalise = 255

    #load images directly into numpy test,train,val matrices.
    def load_matrices(self, re_shape=(80, 120), no_train=False):
        #image_shape = (1,283,283)@channels, height, width
        train_size = int(0.8*self.DATA_SIZE)#=4000
        val_size = int(0.1*self.DATA_SIZE)#=500
        test_size = int(0.1*self.DATA_SIZE)#=500
        
        train_x = numpy.zeros(1)
        train_y = [None]
        val_x = numpy.zeros((val_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        test_x = numpy.zeros((test_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        val_y = [None] * val_size
        test_y = [None] * test_size
        if no_train==False:
            train_x = numpy.zeros((train_size, 1, re_shape[0], re_shape[1]), dtype='float32')
            train_y = [None] * train_size

        split_and_shuffle = range(self.DATA_SIZE)
        random.shuffle(split_and_shuffle)
        #distributes an element and label in either train,val,test sets using specified proportions
        def distribute(element, label, idx):
            new_idx = split_and_shuffle[idx]
            if new_idx < train_size:
                train_x[new_idx] = element
                train_y[new_idx] = label
            elif new_idx < val_size + train_size:
                val_x[new_idx%val_size] = element
                val_y[new_idx%val_size] = label
            else:
                test_x[new_idx%test_size] = element
                test_y[new_idx%test_size] = label

        print("loading sclera matrices..")
        #load the images into our numpy array
        cnt = 0
        for root, dirs, filenames in os.walk(self.DATA_PATH):
            for f in filenames:
                #do not read training images
                if no_train and split_and_shuffle[cnt] < train_size:
                    cnt += 1
                    continue

                img = keras.preprocessing.image.load_img(os.path.join(root,f), grayscale=True)
                img = scipy.misc.imresize(img, re_shape)
                x = keras.preprocessing.image.img_to_array(img, 'th')

                #append image array to training, validation or test set and set the label
                distribute(x, os.path.basename(root), cnt)
                cnt += 1

        train_x /= self.rgb_normalise
        val_x /= self.rgb_normalise
        test_x /= self.rgb_normalise

        return (train_x, train_y, val_x, val_y, test_x, test_y)

    #load images directly into numpy test,train matrices. The training set is meant for use with cross validation
    def load_matrices_cross(self, re_shape=(80, 120)):
        #image_shape = (1,283,283)@channels, height, width
        train_size = int(0.9*self.DATA_SIZE)#=4500
        test_size = int(0.1*self.DATA_SIZE)#=500
        
        train_x = numpy.zeros((train_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        train_y = [None] * train_size
        test_x = numpy.zeros((test_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        test_y = [None] * test_size

        split_and_shuffle = range(self.DATA_SIZE)
        random.shuffle(split_and_shuffle)

        fnames = dict()
        #distributes an element and label in either train,val,test sets using specified proportions
        def distribute(element, label, idx):
            new_idx = split_and_shuffle[idx]
            if new_idx < train_size:
                train_x[new_idx] = element
                train_y[new_idx] = label
            else:
                test_x[new_idx%test_size] = element
                test_y[new_idx%test_size] = label

        print("loading sclera matrices..")
        #load the images into our numpy array
        cnt = 0
        for root, dirs, filenames in os.walk(self.DATA_PATH):
            for f in filenames:
                img = keras.preprocessing.image.load_img(os.path.join(root,f), grayscale=True)
                img = scipy.misc.imresize(img, re_shape)
                x = keras.preprocessing.image.img_to_array(img, 'th')

                #append image array to training, validation or test set and set the label
                distribute(x, os.path.basename(root), cnt)
                if split_and_shuffle[cnt] >= train_size:
                    fnames[os.path.join(root,f)] = cnt
                cnt += 1

        CNNLIB().save_dict(fnames, self.TESTSET_PATH)
        print("normalising matrices..")
        train_x /= self.rgb_normalise
        test_x /= self.rgb_normalise

        return (train_x, train_y, test_x, test_y)

    #Given that the load_matrices_cross has run before, load the test data as determined by that run
    def load_test_cross(self, re_shape=(80, 120)):
        #image_shape = (1,283,283)@channels, height, width
        test_size = int(0.1*self.DATA_SIZE)#=1025
        
        test_x = numpy.zeros((test_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        test_y = [None] * test_size

        fnames = CNNLIB().load_dict(self.TESTSET_PATH)
        fnames = [abs_path for abs_path in fnames]

        print("loading sclera matrices into test set..")
        #load the images into our numpy array
        cnt = 0
        for root, dirs, filenames in os.walk(self.DATA_PATH):
            for f in filenames:
                if os.path.join(root,f) in fnames:
                    img = keras.preprocessing.image.load_img(os.path.join(root,f), grayscale=True)
                    img = scipy.misc.imresize(img, re_shape)
                    x = keras.preprocessing.image.img_to_array(img, 'th')

                    #append image array to test set and set the label
                    test_x[cnt] = x
                    test_y[cnt] = os.path.basename(root)
                    cnt += 1
        test_x /= self.rgb_normalise

        return (test_x, test_y)

    #Given that the load_matrices_cross has run before, load the train data as determined by that run
    def load_train_cross(self, re_shape=(80, 120)):
        #image_shape = (1,283,283)@channels, height, width
        train_size = int(0.9*self.DATA_SIZE)#=1025
        
        train_x = numpy.zeros((train_size, 1, re_shape[0], re_shape[1]), dtype='float32')
        train_y = [None] * train_size

        fnames = CNNLIB().load_dict(self.TESTSET_PATH)
        fnames = [abs_path for abs_path in fnames]

        print("loading sclera matrices into train set..")
        #load the images into our numpy array
        cnt = 0
        for root, dirs, filenames in os.walk(self.DATA_PATH):
            for f in filenames:
                if not os.path.join(root,f) in fnames:
                    img = keras.preprocessing.image.load_img(os.path.join(root,f), grayscale=True)
                    img = scipy.misc.imresize(img, re_shape)
                    x = keras.preprocessing.image.img_to_array(img, 'th')

                    #append image array to training set and set the label
                    train_x[cnt] = x
                    train_y[cnt] = os.path.basename(root)
                    cnt += 1
        train_x /= self.rgb_normalise

        return (train_x, train_y)
