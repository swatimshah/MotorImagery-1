import numpy
from tensorflow.random import set_seed
from numpy.random import seed

# setting the seed
seed(1)
set_seed(1)

data_22_11_28 = numpy.loadtxt('EEGData_For_Online_22.11.03_Filtered_pca.csv', delimiter=',')
data_22_11_24 = numpy.loadtxt('EEGData_For_Online_22.11.11_Filtered_pca.csv', delimiter=',')
data_22_11_11 = numpy.loadtxt('EEGData_For_Online_22.11.24_Filtered_pca.csv', delimiter=',')
data_22_11_03 = numpy.loadtxt('EEGData_For_Online_22.11.28_Filtered_pca.csv', delimiter=',')
data_22_12_13 = numpy.loadtxt('EEGData_For_Online_22.12.13_Filtered_pca.csv', delimiter=',')
data_22_12_15 = numpy.loadtxt('EEGData_For_Online_22.12.15_Filtered_pca.csv', delimiter=',')


complete_data = numpy.concatenate((data_22_11_28, data_22_11_24, data_22_11_11, data_22_11_03, data_22_12_13, data_22_12_15), axis=0)	

numpy.savetxt('complete_pca_data.csv', complete_data, delimiter=',')	
print(complete_data.shape)

complete_data_printed = complete_data.transpose()
numpy.savetxt('complete_pca_data_printed.csv', complete_data_printed, delimiter=',')