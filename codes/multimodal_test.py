import keras
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from nn_lib import CNNLIB
from sclera250 import Sclera
from words250 import WordImages

sclera250 = Sclera()
words250 = WordImages()
nn_lib = CNNLIB()

(label2index, index2label, nb_classes) = nn_lib.load_labeling()
shape_image = (80, 120)
shape_words = (28, 184)

basename_models_iw = "../neural_models/multimodal/nodropout/image+word"#/nodropout
basename_models_ii = "../neural_models/multimodal/nodropout/image+image"#/nodropout
basename_models_ww = "../neural_models/multimodal/nodropout/word+word"#/nodropout

def format_data(input_x1, input_y1, input_x2, input_y2):
    """Helper function returning merged data, merged labels and the train/validation split"""
    # (multi_data, multi_y) = nn_lib.mergesets(input_x2, input_x1, input_y2, input_y1, proportion = 0.4)
    (multi_data, multi_y) = nn_lib.concatsets(input_x2, input_x1, input_y2, input_y1)
    #k_folds = 10
    #skf = StratifiedKFold(multi_y, n_folds=k_folds, shuffle=True, random_state=42)
    multi_y = nn_lib.format_labels(multi_y, label2index, nb_classes)

    #obtain validation data for first fold (incorrect method. Use appropriate fold during training instead)
    #for i, (train_set, val_set) in enumerate(skf):
    #    return ([multi_data[1][val_set], multi_data[0][val_set]], multi_y[val_set])

    return ([multi_data[1], multi_data[0]], multi_y)

print "loading models.."
scores_iw, scores_iw_images, scores_iw_words, scores_ii, scores_ii_images, scores_ii_words, scores_ww, scores_ww_images, scores_ww_words  = ([] for i in range(9))
models_iw, models_ii, models_ww = ([] for i in range(3))
for i in range(10):
    models_iw.append(keras.models.load_model(basename_models_iw + str(i) + ".h5"))
    models_ii.append(keras.models.load_model(basename_models_ii + str(i) + ".h5"))
    models_ww.append(keras.models.load_model(basename_models_ww + str(i) + ".h5"))

print "Loading word/image data"
(imgs_x, imgs_y) = sclera250.load_test_cross(re_shape = shape_image)
(words_x, words_y) = words250.load_test_cross(re_shape = shape_words)
(multi_data, multi_y) = format_data(imgs_x, imgs_y, words_x, words_y)
print "Loading image/image data"
(img2_data, img2_y) = format_data(imgs_x, imgs_y, imgs_x, imgs_y)
print "Loading word/word data"
(wrds2_data, wrds2_y) = format_data(words_x, words_y, words_x, words_y)

# print "mdat0sh, mdat1sh" + str((multi_data[0].shape, multi_data[1].shape))
# print "wrds0, 0" + str((imgs_x.shape, np.zeros((imgs_x.shape[0],) + multi_data[1].shape[1:]).shape))
# print "0, images" + str((np.zeros((words_x.shape[0],) + multi_data[0].shape[1:]).shape, words_x.shape))

classes_ii = nn_lib.format_labels(imgs_y, label2index, nb_classes)
classes_ww = nn_lib.format_labels(words_y, label2index, nb_classes)

def zeros(length_of_, shape_of_):
    "return a matrix with shape dictated by the length of first matrix and shape of elements in the second"
    return np.zeros((length_of_.shape[0], ) + shape_of_.shape[1:])

#Testing also a single modality where the other remains zero shows how well the network uses both input modalities.
#(single mixed training, testing modality case dependent)
for i in range(10):
    scores_ii.append(models_ii[i].evaluate(img2_data, img2_y, batch_size=25)[1])
    scores_ii_images.append(models_ii[i].evaluate([imgs_x, zeros(imgs_x, img2_data[1])], classes_ii, batch_size=25)[1])
    scores_ii_words.append(models_ii[i].evaluate([zeros(imgs_x, img2_data[0]), imgs_x], classes_ii, batch_size=25)[1])

    scores_iw.append(models_iw[i].evaluate(multi_data, multi_y, batch_size=25)[1])#[resize(imgs, len(words), words))] as input
    scores_iw_images.append(models_iw[i].evaluate([imgs_x, zeros(imgs_x, multi_data[1])], classes_ii, batch_size=25)[1])
    scores_iw_words.append(models_iw[i].evaluate([zeros(words_x, multi_data[0]), words_x], classes_ww, batch_size=25)[1])

    scores_ww.append(models_ww[i].evaluate(wrds2_data, wrds2_y, batch_size=25)[1])
    scores_ww_images.append(models_ww[i].evaluate([words_x, zeros(words_x, wrds2_data[1])], classes_ww, batch_size=25)[1])
    scores_ww_words.append(models_ww[i].evaluate([zeros(words_x, wrds2_data[0]), words_x], classes_ww, batch_size=25)[1])

print "2ximage classifier/multi input: "   + str(scores_ii)
print "2ximage classifier/image input: "   + str(scores_ii_images)
print "2ximage classifier/word input: "    + str(scores_ii_words)
print "2xword classifier/multi input: "   + str(scores_ww)
print "2xword classifier/image input: "   + str(scores_ww_images)
print "2xword classifier/word input: "    + str(scores_ww_words)
print "multi classifier/multi input: "   + str(scores_iw)
print "multi classifier/image input: "   + str(scores_iw_images)
print "multi classifier/word input: "    + str(scores_iw_words)