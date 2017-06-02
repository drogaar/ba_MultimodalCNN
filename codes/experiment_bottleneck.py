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
for b in reversed(range(18)):
    nb_representation = b + 1
    nb_kernels = 6

    for i, (train_set,val_set) in enumerate(skf):
        classifier_ordinary = None#ensure empty
        classifier_ordinary = nn_lib.encoder2classifier(nn_lib.define_encoder(nb_kernels, nb_representation))

        nn_lib.train_net(classifier_ordinary, data_x[train_set], data_y[train_set], data_x[val_set], data_y[val_set],
                            "img2class_k"+str(nb_kernels) + "b" + str(nb_representation) + "i" + str(i), epochs=12, path="../neural_models/by_representation/")

        #score metrics : [loss, accuracy]
        score = classifier_ordinary.evaluate(data_x[val_set],data_y[val_set], batch_size=25)
        output.append({'k':nb_kernels,'b':nb_representation,'i':i,'score':score[1]})
        
for result in output:
    print result
nn_lib.jsondump(output, "output_representation.json")
print "Training complete =D"





