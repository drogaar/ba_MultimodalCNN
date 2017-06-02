import keras
#import keras.backend as K
from sclera250 import Sclera
from words250 import WordImages
#from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from nn_lib import CNNLIB

nn_lib = CNNLIB()

#path configuration. This experiment can be used for path_image/word with or without dropout or with different shapes
path_image = "../neural_models/unimodal/img2class_6146.h5"#nodropout: "../neural_models/unimodal/nodropout/img2class_678.h5"
path_word = "../neural_models/unimodal/img2class_121108.h5"#nodropiut: "../neural_models/unimodal/nodropout/word2class_12552.h5"
path_neuralmodels = "../neural_models/multimodal/dropout/" #"../neural_models/multimodal/no_dropout"
path_output = "output_multimodal_29mo_dropout.json"
path_history = "history_multimodal_29mo_dropoutjson"

print "Defining labels"
(label2index, index2label, nb_classes) = nn_lib.load_labeling()
output = []
history = []

def train_family(class1path, class2path, multi_data, multi_labels, skf, shapes, name="image+word", seqnum1=1, seqnum2=2):
    """Trains the multimodal classifier on 10 folds.
    classpaths should the contain the paths to unimodal classifiers.
    multi_data and multi_labels reference the required dataset
    train_set and val_set iterate the dataset such that the data is split in valid training/validation sets.
    The seqnum's should be given the sequential layer number, as assigned by default keras.
    name is used merely for saving the resulting networks
    """
    for i, (train_set,val_set) in enumerate(skf):
        classifier_multi = None#ensure empty
        img_encoder = nn_lib.reuse_model(class1path, 'sequential_' + str(2*i + seqnum1))
        wrd_encoder = nn_lib.reuse_model(class2path, 'sequential_' + str(2*i + seqnum2))
        classifier_multi = nn_lib.encoders2classifier(img_encoder, wrd_encoder, shapes)
        if i == 0: classifier_multi.summary()

        hist_it = nn_lib.train_net(classifier_multi, [medium[train_set] for medium in multi_data], multi_y[train_set], 
                       [medium[val_set] for medium in multi_data], multi_y[val_set], name + str(i), epochs=150, path=path_neuralmodels,
                        earlystop=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.002, patience=2, mode='auto')])

        score = classifier_multi.evaluate( [medium[val_set] for medium in multi_data], multi_y[val_set], batch_size=25)
        output.append({'type':name,'i':i, 'score':score[1]})
        history.append({'type':name, 'i':i, 'history':hist_it.history["val_acc"]})

def format_data(input_x1, input_y1, input_x2, input_y2):
    """Helper function returning merged data, merged labels and the train/validation split"""
    (multi_data, multi_y) = nn_lib.mergesets(input_x2, input_x1, input_y2, input_y1, proportion = 0.4)
    skf = StratifiedKFold(multi_y, n_folds=k_folds, shuffle=True, random_state=42)
    multi_y = nn_lib.format_labels(multi_y, label2index, nb_classes)
    return ([multi_data[1], multi_data[0]], multi_y, skf)

#load data
shape_image = (80, 120)
shape_words = (28,184)
sclera250 = Sclera()
words250 = WordImages()
k_folds = 10

print("training image+word...")
(imgs_x, imgs_y) = sclera250.load_train_cross(re_shape = shape_image)
(words_x, words_y) = words250.load_train_cross(re_shape = shape_words)
(multi_data, multi_y, skf) = format_data(imgs_x, imgs_y, words_x, words_y)
print "1: " + str(multi_data)
print "1: " + str(multi_data[0].shape)
train_family(path_image, path_word, multi_data, multi_y, skf, (shape_image, shape_words), "image+word", 1, 2)
print("training image+image...")
(multi_data, multi_y, skf) = format_data(imgs_x, imgs_y, imgs_x, imgs_y)
train_family(path_image, path_image, multi_data, multi_y, skf, (shape_image, shape_image), "image+image", 21, 22)
print("training word+word...")
(multi_data, multi_y, skf) = format_data(words_x, words_y, words_x, words_y)
train_family(path_word, path_word, multi_data, multi_y, skf, (shape_words, shape_words), "word+word", 41, 42)
           
print "Training complete =D"
nn_lib.jsondump(output, path_output)
nn_lib.jsondump(history, path_history)