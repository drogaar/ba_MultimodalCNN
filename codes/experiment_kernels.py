import keras
import keras.backend as K
from sclera250 import Sclera
from sklearn.cross_validation import StratifiedKFold
from nn_lib import CNNLIB

nn_lib = CNNLIB()
#load data
new_shape = (80,120)
sclera250 = Sclera()
(data_x, data_y) = sclera250.load_train_cross(re_shape = new_shape)

print "Defining labels"
(label2index, index2label, nb_classes) = nn_lib.load_labeling()

k_folds = 10
skf = StratifiedKFold(data_y, n_folds=k_folds, shuffle=True, random_state=42)
data_y = nn_lib.format_labels(data_y, label2index, nb_classes)

print("training network...")
output = []
history = []
for k in reversed(range(10)):#range(10):
    nb_representation = 7
    nb_kernels = 2*k + 2

    for i, (train_set,val_set) in enumerate(skf):
        classifier_ordinary = None#ensure empty
        classifier_ordinary = nn_lib.encoder2classifier(nn_lib.define_encoder(nb_kernels, nb_representation))

        hist_it = nn_lib.train_net(classifier_ordinary, data_x[train_set], data_y[train_set], data_x[val_set], data_y[val_set],
                            "img2class_k"+str(nb_kernels) + "_b" + str(nb_representation) + "_i" + str(i), epochs=40, path="../neural_models/by_kernel/may27/", 
                            earlystop=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.005, patience=2, mode='auto')])
        #14 epochs, fixed used earlier with nn_lib.train_net
        #new approach: 25 epochs? patience 3 
        #also explain effect earlystopping:
        # higher accuracy for small models than the higher accurcy for complex models.

        #score metrics : [loss, accuracy]
        score = classifier_ordinary.evaluate(data_x[val_set],data_y[val_set], batch_size=25)
        output.append({'k':nb_kernels,'b':nb_representation,'i':i,'score':score[1]})
        history.append({'k':nb_kernels, 'i':i, 'history':hist_it.history["val_acc"]})
        
for result in output:
    print result
nn_lib.jsondump(output, "output_kernels_27may.json")
nn_lib.jsondump(history, "history_kernels_27may.json")
print "Training complete =D"





