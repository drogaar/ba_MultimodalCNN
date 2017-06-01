import keras
import csv
import json
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle

class CNNLIB():
    """
    This class provides random useful methods for use in training scripts.
    """

    labels_path = "../data/labeling/"
    POOLING = (2,2)
    CLASSES = 250
    shape_image=(1,80,120)
    shape_word=(1,28,184)

    def jsondump(self, data, name="output"):
        with open('../log/' + name, 'w') as fout:
            json.dump(data, fout)

    def jsonload(self, name="output"):
        with open('../log/' + name, 'r') as fin:
            data = json.load(fin)
        return data

    def load_dict(self, loc="../data/labeling/lbl2idx.csv"):
        """Load a dictionary from the specified csv file saved using save_dict"""
        data = dict()
        with open(loc) as f:
            reader=csv.DictReader(f, ['key','value'])
            for row in reader:
                data[row['key']] = row['value']
        return data

    def save_dict(self, dict_data, loc="../data/labeling/lbl2idx.csv"):
        """Save a dictionary in a specified csv file"""
        with open(loc, 'w') as f:
            w = csv.DictWriter(f, ['key','value'])
            for key in dict_data:
                w.writerow({'key':key, 'value':dict_data[key]})

    def define_labeling(self, data):
        """Given all labels for a dataset, this defines a one-to-one relation between the labels and a set of integer numbers.
        The relation is saved by the function from label to integer and function from integer to label pair and can be 
        retrieved using the load_labeling function.
        """
        labels = set(data)
        nb_classes = len(labels)

        label2index = dict(zip(labels,range(nb_classes)))
        index2label = {}
        for k,v in label2index.iteritems():
            index2label[v] = k

        self.save_dict(label2index, self.labels_path + "lbl2idx.csv")
        self.save_dict(index2label, self.labels_path + "idx2lbl.csv")
        return (label2index, index2label, nb_classes)

    def load_labeling(self):
        """Retrieves the function mapping labels to integers and the one mapping integers to labels, as defined earlier by define_labeling"""
        label2index = self.load_dict(self.labels_path + "lbl2idx.csv")
        idx2lbl = self.load_dict(self.labels_path + "idx2lbl.csv")

        #convert string indices to integers
        index2label = dict()
        for key in label2index:
            label2index[key] = int(label2index[key])
        for key in idx2lbl:
            index2label[int(key)] = idx2lbl[key]
        
        return (label2index, index2label, len(label2index))

    def mergesets(self, data_x_1, data_x_2, data_y_1, data_y_2, proportion = 0.6):
        """merge 2 datasets into a single where proportion indicates the percentage of training instances where both samples from 1 and 2 are fed.
        The remaining percentage will be split equally between feeding data from set 1 and from set 2.
        assumes data_x_1 to be the smallest

        To keep the dataset unbiased, no modality should contain more classes or examples than its countermodality. 
        As such, the number of samples is equal for both modalities.
        Moreover, any class in only modality but not in the other is truncated from the dataset.
        """

        assert(len(data_x_1) <= len(data_x_2))

        def gather(labels_small, labels_large):
            """Returns tuple of 2 lists where:
            permutation = items x : x in labels_small and x in      labels_large
            truncate    = items x : x in labels_small and x not in  labels_large
            together they allow picking classes contained in both modalities equally"""
            permutation = []
            truncate_idxs = []
            for idx_small in range(len(labels_small)):
                for idx_large in range(len(data_y_2)):
                    if data_y_1[idx_small] == data_y_2[idx_large] and not idx_large in permutation:
                        permutation.append(idx_large)
                        break
                else:
                    truncate_idxs.append(idx_small)
            return (permutation, truncate_idxs)

        #truncate classes in small but not in large
        (_, truncate_idxs) = gather(data_y_1, data_y_2)
        print "truncated original size data: " + str(len(data_x_1))
        data_x_1 = np.delete(data_x_1, truncate_idxs, 0)
        data_y_1 = list(np.delete(data_y_1, truncate_idxs))
        print "to : " + str(len(data_x_1))

        minlen = len(data_x_1)
        unimodal_len = int((1-proportion) * minlen)
        full_len = minlen + unimodal_len

        #shuffle class distribution in data before combining modalities
        data_x_1, data_y_1 = shuffle(data_x_1, data_y_1)
        data_x_2, data_y_2 = shuffle(data_x_2, data_y_2)

        #obtain indices of images in x2 corresponding to the classes in x1
        #this operation truncates images from the largest matrix when not present in the small dataset.
        (permutation, _) = gather(data_y_1, data_y_2)
        assert(minlen == len(permutation)) #each label in the small dataset should have a corresponding item in the second dataset
        
        #allocate multi_modal matrices
        data_x_1.resize((full_len,) + data_x_1.shape[1:])
        data_y_1 = data_y_1 + [0 for i in range(full_len - minlen)]
        adjusted_x_2 = np.zeros((full_len,) + data_x_2.shape[1:])
        adjusted_y_2 = ["" for i in range(full_len)]

        #fill in multi modal data. First both. Then the small set only. then the largest only.
        for cnt, permute in enumerate(permutation):
            if cnt < proportion * minlen:#both modes <2700
                adjusted_x_2[cnt, :, :, :] = data_x_2[permute, :, :, :]
                adjusted_y_2[cnt] = data_y_2[permute]
            else:#mat1 only <2700, mat2 only 2700 + 1800 = 4500 < 6300
                adjusted_x_2[cnt + unimodal_len, :, :, :] = data_x_2[permute, :, :, :]
                adjusted_y_2[cnt + unimodal_len] = data_y_2[permute]
                data_y_1[cnt + unimodal_len] = data_y_2[permute]

        #shuffle media gaps in data
        data_x_1, adjusted_x_2, data_y_1, adjusted_y_2 = shuffle(data_x_1, adjusted_x_2, data_y_1, adjusted_y_2)
        return ([data_x_1, adjusted_x_2], data_y_1)

    def concatsets(self, data_x_1, data_x_2, data_y_1, data_y_2):
        """Matches samples from set 1 with samples from set 2, similar to what mergesets does.
        However, this function does not introduce unimodal samples in the data."""

        assert(len(data_x_1) <= len(data_x_2))

        def gather(labels_small, labels_large):
            """Returns tuple of 2 lists where:
            permutation = items x : x in labels_small and x in      labels_large
            truncate    = items x : x in labels_small and x not in  labels_large
            together they allow picking classes contained in both modalities equally"""
            permutation = []
            truncate_idxs = []
            for idx_small in range(len(labels_small)):
                for idx_large in range(len(data_y_2)):
                    if data_y_1[idx_small] == data_y_2[idx_large] and not idx_large in permutation:
                        permutation.append(idx_large)
                        break
                else:
                    truncate_idxs.append(idx_small)
            return (permutation, truncate_idxs)

        #truncate classes in small but not in large
        (_, truncate_idxs) = gather(data_y_1, data_y_2)
        print "truncated original size data: " + str(len(data_x_1))
        data_x_1 = np.delete(data_x_1, truncate_idxs, 0)
        data_y_1 = list(np.delete(data_y_1, truncate_idxs))
        print "to : " + str(len(data_x_1))

        minlen = len(data_x_1)

        #shuffle class distribution in data before combining modalities
        data_x_1, data_y_1 = shuffle(data_x_1, data_y_1)
        data_x_2, data_y_2 = shuffle(data_x_2, data_y_2)

        #allocate the second matrix
        adjusted_x_2 = np.zeros((minlen,) + data_x_2.shape[1:])
        adjusted_y_2 = ["" for i in range(minlen)]

        #obtain indices of images in x2 corresponding to the classes in x1
        #this operation truncates images from the largest matrix when not present in the small dataset.
        (permutation, _) = gather(data_y_1, data_y_2)
        assert(minlen == len(permutation)) #each label in the small dataset should have a corresponding item in the second dataset

        #combine both modalities
        for cnt, permute in enumerate(permutation):
                adjusted_x_2[cnt, :, :, :] = data_x_2[permute, :, :, :]
                adjusted_y_2[cnt] = data_y_2[permute]

        return ([data_x_1, adjusted_x_2], data_y_1)

    def define_encoder(self, nb_kernels=6, nb_bottleneck=14, dropout=False, word_images=False):
        """Define a network as used for the research"""
        img_shape = self.shape_image
        if word_images: img_shape = self.shape_word
        print("Defining network with " + str(nb_kernels) + " kernels and " + str(nb_bottleneck) + " nodes in the output layer.")
        encoder = keras.models.Sequential()
        if dropout:
            encoder.add(keras.layers.core.Dropout(0.2, input_shape=img_shape))
            encoder.add(keras.layers.Convolution2D(nb_kernels/2, 3,3,activation='relu',border_mode='same'))
        else:
            encoder.add(keras.layers.Convolution2D(nb_kernels/2, 3,3,activation='relu',border_mode='same', input_shape=img_shape))
        encoder.add(keras.layers.MaxPooling2D(self.POOLING, border_mode='same'))
        if dropout: encoder.add(keras.layers.core.Dropout(0.1))

        encoder.add(keras.layers.Convolution2D(nb_kernels, 3,3,activation='relu',border_mode='same'))
        encoder.add(keras.layers.MaxPooling2D(self.POOLING, border_mode='same'))
        if dropout: encoder.add(keras.layers.core.Dropout(0.1))

        encoder.add(keras.layers.Convolution2D(nb_kernels, 3,3,activation='relu',border_mode='same'))
        encoder.add(keras.layers.MaxPooling2D(self.POOLING, border_mode='same'))
        if dropout: encoder.add(keras.layers.core.Dropout(0.1))

        encoder.add(keras.layers.Convolution2D(nb_kernels, 3,3,activation='relu',border_mode='same'))#use nb_kernels kernels instead
        encoder.add(keras.layers.core.Flatten())
        if dropout: encoder.add(keras.layers.core.Dropout(0.1))

        encoder.add(keras.layers.Dense(nb_bottleneck, activation='relu'))
        if dropout: encoder.add(keras.layers.core.Dropout(0.5))

        return encoder

    def reuse_model(self, path, sequential_name="sequential_1"):
        """Given a saved classifier, this function strips the encoder from the classifier.
        The encoder layers are then frozen such that a new classifier can reuse the encodings."""
        classifier = keras.models.load_model(path)

        #retrieve the representational parts
        layer_dict = dict([(layer.name,layer) for layer in classifier.layers])
        print("layer_dict keys: ",layer_dict.keys())
        seq = layer_dict[sequential_name]

        #disable training, recompile to start effect (keeps weights)
        for layer in seq.layers:
            layer.trainable = False
        seq.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        return seq

    def encoder2classifier(self, encoder, word_images=False):
        """Given an encoder, this function returns its classifier."""
        img_shape = self.shape_image
        if word_images: img_shape = self.shape_word
        input_img = keras.layers.Input(shape=img_shape)

        encoded_img1 = encoder(input_img)
        classified = keras.layers.Dense(self.CLASSES, activation='softmax')(encoded_img1)

        classifier = keras.models.Model(input_img, output=classified)
        classifier.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

    def encoders2classifier(self, encoder1, encoder2, shapes):
        """Given 2 encoders, this function returns a classifier for both of the inputs"""
        input1 = keras.layers.Input(shape=(1,) + shapes[0])
        input2 = keras.layers.Input(shape=(1,) + shapes[1])

        encoder1 = encoder1(input1)
        encoder2 = encoder2(input2)

        #multi_modal classification layer
        merged = keras.layers.merge([encoder1, encoder2], mode='concat')
        merged = keras.layers.Dense(self.CLASSES, activation='softmax')(merged)

        classifier_multi = keras.models.Model([input1, input2], output=merged)
        classifier_multi.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier_multi

    def train_net(self, network, train_x, train_y, val_x, val_y, identifier="temporary_net", epochs=10, path="../neural_models/other/", earlystop=False):
        """Trains and saves a network. Supply earlystop with the list of callbacks to use."""
        callbacks_list = []
        if earlystop: callbacks_list = earlystop
        history = network.fit(train_x,
                train_y,
                nb_epoch=epochs,
                batch_size=10,
                shuffle=True,
                validation_data=(val_x, val_y),
                callbacks = callbacks_list
                )
        print("saving network.. " + identifier + ".")
        network.save(path + identifier + ".h5")
        return history
        
    def format_labels(self, target_labels, label2index, nb_labels):
        """Given a mapping from labels to index, this converts a list of target labels to a one-hot matrix where each of the rows is 1 for only 1 of the classes."""
        return np_utils.to_categorical([label2index[i] for i in target_labels], nb_labels)
