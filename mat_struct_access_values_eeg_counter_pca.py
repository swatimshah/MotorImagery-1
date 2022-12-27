import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA
from tensorflow.random import set_seed
from numpy.random import seed
from sklearn.preprocessing import RobustScaler


def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)


# setting the seed
seed(1)
set_seed(1)

final_input_data = numpy.empty([0, 4096])

myKeys = loadmat("EEGData_unit_22.12.15_Filterd_1_Reshaped_1.mat")
print(myKeys)
eegData = myKeys['EEGData_unit']
eegDataAllSamples = eegData['Data']
eegDataAllLabels = eegData['Labels']
print(eegDataAllSamples.shape)
print(eegDataAllLabels.shape)

push_input = eegDataAllLabels == 'Push'
eegDataAllLabels[push_input] = 2
pull_input = eegDataAllLabels == 'Pull'
eegDataAllLabels[pull_input] = 1
no_input = eegDataAllLabels == 'Nothing'
eegDataAllLabels[no_input] = 0

for i in range (300):
	eegData_orig_temp = eegDataAllSamples[i].EEG_Data
	print(eegData_orig_temp.shape)
	print("Values bigger than 1.0 =", eegData_orig_temp[eegData_orig_temp > 1.0])

	my_pca = PCA(n_components=64, random_state=2)
	my_pca.fit(eegData_orig_temp)
	print(my_pca.components_.shape)

	input_to_nn = my_pca.components_.flatten().reshape(1, 4096)
	final_input_data = numpy.append(final_input_data, input_to_nn, axis=0)

wholeData = numpy.append(final_input_data, eegDataAllLabels.reshape(len(eegDataAllLabels), 1), axis=1)

savetxt('EEGData_For_Online_22.12.15_Filtered_pca.csv', wholeData, delimiter=',')
print(wholeData.shape)

wholeData_printed = wholeData.transpose()
numpy.savetxt('wholeData_filtered_printed_22.12.15_pca.csv', wholeData_printed[0:4096, 0:300], delimiter=',')	
