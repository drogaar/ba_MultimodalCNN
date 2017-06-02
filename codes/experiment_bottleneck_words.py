import keras
import keras.backend as K
from words250 import WordImages
from sklearn.cross_validation import StratifiedKFold
from nn_lib import CNNLIB

nn_lib = CNNLIB()
#load data
new_shape = (28,184)
words250 = WordImages()
(data_x, data_y) = words250.load_train_cross(re_shape = new_shape)

print "Defining labels"
(label2index, index2label, nb_classes) = nn_lib.load_labeling()

k_folds = 10
skf = StratifiedKFold(data_y, n_folds=k_folds, shuffle=True, random_state=42)
data_y = nn_lib.format_labels(data_y, label2index, nb_classes)

print("training network...")
output = []
for b in reversed(range(22)):#reversed(range(13)):#35:2b+2: 65/5=13, 64/4=16, 66/3=22
    nb_representation = 3*b + 3
    nb_kernels = 12

    for i, (train_set,val_set) in enumerate(skf):
        classifier_ordinary = None#ensure empty
        encoder = nn_lib.define_encoder(nb_kernels, nb_representation, word_images=True)
        classifier_ordinary = nn_lib.encoder2classifier(encoder, word_images=True)

        nn_lib.train_net(classifier_ordinary, data_x[train_set], data_y[train_set], data_x[val_set], data_y[val_set],
                            "img2class_k"+str(nb_kernels) + "b" + str(nb_representation) + "i" + str(i), epochs=20, path="../neural_models/by_representation_word/")
        #15 epochs estimate earlystop=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.005, patience=2, mode='auto')]
        #20
        #23~?

        #score metrics : [loss, accuracy]
        score = classifier_ordinary.evaluate(data_x[val_set],data_y[val_set], batch_size=25)
        output.append({'k':nb_kernels,'b':nb_representation,'i':i,'score':score[1]})
        
for result in output:
    print result
nn_lib.jsondump(output, "output_representation_words.json")
print "Training complete =D"





