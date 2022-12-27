from numpy import loadtxt
from numpy import savetxt
import numpy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow
from keras.layers import LSTM, Activation, Bidirectional
from keras.layers import TimeDistributed

# setting the seed
seed(1)
set_seed(1)

index1 = 1800

# load the data from the csv file
orig_epochs = loadtxt('complete_pca_data.csv', delimiter=',')
print(orig_epochs.shape)

# shuffle the training data
numpy.random.seed(2) 
numpy.random.shuffle(orig_epochs)
print(orig_epochs.shape)

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(orig_epochs[0:index1, :], orig_epochs[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)


print("********************")


# augment train data
X_total_1 = numpy.append(X_train_tmp, X_train_tmp, axis=0)
X_total = numpy.append(X_total_1, X_train_tmp, axis=0)
print(X_total.shape)
print(X_total[:, -1].astype(int).ravel()) 


print("-------------------")



# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].astype(int).ravel())
print("After OverSampling, counts of label '2': {}".format(sum(Y_train_keep == 2)))
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))
print(X_train_keep.shape)

numpy.random.shuffle(X_train_keep)

#=======================================
 
# Data Pre-processing - scale data using robust scaler

Y_train = X_train_keep[:, -1]
Y_test = X_test_tmp[:, -1]

print(Y_train)
print(Y_test)

input = X_train_keep[:, 0:1920]
testinput = X_test_tmp[:, 0:1920]

#=====================================

# Model configuration

input = input.reshape(len(input), 30, 64)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 30, 64)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

# Create the model - Augmented Twice
weights = 0.045
negative_weights = -0.045

L2_Regularization = 0.00022

# Create the model
model=Sequential()
model.add(Conv1D(filters=48, kernel_size=7, kernel_regularizer=L2(L2_Regularization), bias_regularizer=L2(L2_Regularization), activity_regularizer = L2(L2_Regularization), kernel_constraint=min_max_norm(min_value=negative_weights, max_value=weights), padding='valid', activation='relu', strides=2, input_shape=(64, 30)))
model.add(Conv1D(filters=48, kernel_size=5, kernel_regularizer=L2(L2_Regularization), bias_regularizer=L2(L2_Regularization), activity_regularizer = L2(L2_Regularization), kernel_constraint=min_max_norm(min_value=negative_weights, max_value=weights), padding='valid', activation='relu', strides=1))
model.add(AveragePooling1D(pool_size=3))
model.add(Dropout(0.3))
model.add(Conv1D(filters=48, kernel_size=5, kernel_regularizer=L2(L2_Regularization), bias_regularizer=L2(L2_Regularization), activity_regularizer = L2(L2_Regularization), kernel_constraint=min_max_norm(min_value=negative_weights, max_value=weights), padding='valid', activation='relu', strides=1))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()

# Compile the model  
adam = Adam(learning_rate=0.000045)
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=600)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(input, Y_train, batch_size=32, epochs=200, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])


# evaluate the model
predict_y = model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


#==================================

model.save("model_conv1d.h5")

# load the best model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluate the model
predict_y = saved_model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


#==================================

# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()


