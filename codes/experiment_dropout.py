import keras
#import keras.backend as K
from sclera250 import Sclera
from sklearn.cross_validation import StratifiedKFold
from nn_lib import CNNLIB

nn_lib = CNNLIB()
print "Defining labels"
(label2index, index2label, nb_classes) = nn_lib.load_labeling()

#load data
new_shape = (80,120)
sclera250 = Sclera()
k_folds = 10


print("training image+word...")
(imgs_x, imgs_y) = sclera250.load_train_cross(re_shape = new_shape)
skf = StratifiedKFold(imgs_y, n_folds=k_folds, shuffle=True, random_state=42)
imgs_y = nn_lib.format_labels(imgs_y, label2index, nb_classes)

def train_eval(classifier, trainset, valset, name, epochs, type="ordinary", it=0):
    hist_it = nn_lib.train_net(   classifier, imgs_x[trainset], imgs_y[trainset], imgs_x[valset], imgs_y[valset], identifier=name, epochs=epochs, path="../neural_models/dropout/",
                        earlystop=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0005, patience=2, mode='auto')])
    score = classifier.evaluate(imgs_x[val_set],imgs_y[val_set], batch_size=25)

    output.append({'type':type,'i':it,'score':score[1]})
    history.append({'type':type, 'i':it, 'history':hist_it.history["val_acc"]})

output = []
history = []
for i, (train_set, val_set) in enumerate(skf):
    class_ordinary = nn_lib.encoder2classifier(nn_lib.define_encoder(nb_kernels=6, nb_bottleneck=14, dropout=False))
    class_dropout = nn_lib.encoder2classifier(nn_lib.define_encoder(nb_kernels=6, nb_bottleneck=14, dropout=True))

    train_eval(class_ordinary, train_set, val_set, "ordinary"+str(i), 40, type="ordinary", it=i)
    train_eval(class_dropout, train_set, val_set, "dropout"+str(i), 40, type="dropout", it=i)

    #use path attained from the experiment above
    class_reused = nn_lib.encoder2classifier(nn_lib.reuse_model("../neural_models/dropout/ordinary" + str(i) + ".h5", sequential_name="sequential_" + str(4*i+3)))
    class_dropuse = nn_lib.encoder2classifier(nn_lib.reuse_model("../neural_models/dropout/dropout" + str(i) + ".h5", sequential_name="sequential_" + str(4*i+4)))

    train_eval(class_reused, train_set, val_set, "reused"+str(i), 40, type="reused", it=i)
    train_eval(class_dropuse, train_set, val_set, "dropuse"+str(i), 40, type="dropuse", it=i)
    # if i > 0: break;

for result in output:
    print result
nn_lib.jsondump(output, "output_dropout_31march.json")
nn_lib.jsondump(history, "history_dropout_31march.json")

print "Training complete =D"
