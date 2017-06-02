import keras
import keras.backend as K
from words250 import WordImages
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold
from nn_lib import CNNLIB

nn_lib = CNNLIB()
#load data
shape_words = (28,184)
words250 = WordImages()
(data_x, data_y) = words250.load_train_cross(re_shape = shape_words)

print "Defining labels"
(label2index, index2label, nb_classes) = nn_lib.load_labeling()

k_folds = 10
skf = StratifiedKFold(data_y, n_folds=k_folds, shuffle=True, random_state=42)
data_y = nn_lib.format_labels(data_y, label2index, nb_classes)

print("training network...")
output = []
epochCtr = []
#parameter config
nb_representation = 55#110
nb_kernels = 12
use_dropout = False#True
path_output = "output_words_nodropout_2jun.json"
path_models = "BAD_PATH"#"../neural_models/unimodal/nodropout/"

for i, (train_set,val_set) in enumerate(skf):
    classifier_ordinary = None#ensure empty
    classifier_ordinary = nn_lib.encoder2classifier(nn_lib.define_encoder(nb_kernels, nb_representation, dropout=use_dropout, word_images=True), word_images=True)
    classifier_ordinary.summary()

    hist_it = nn_lib.train_net(classifier_ordinary, data_x[train_set], data_y[train_set], data_x[val_set], data_y[val_set],
                        "word2class_"+str(nb_kernels) + str(nb_representation) + str(i), epochs=150, path=path_models,
                        earlystop=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.002, patience=2, mode='auto')])

    #score metrics : [loss, accuracy]
    score = classifier_ordinary.evaluate(data_x[val_set],data_y[val_set], batch_size=25)
    output.append({'k':nb_kernels,'b':nb_representation,'i':i,'score':score[1]})
    epochCtr.append(len(hist_it.history["val_acc"]))
        
for result in output:
    print result
print epochCtr #[23, 32, 25, 32, 27, 25, 25, 24, 24, 24]

nn_lib.jsondump(output, path_output)
print "Training complete =D"





