import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
import random
#%%
# Test code to make sure things are working
sample_size = 220500
# scipy.spectrogram() is outputting a windows of size 129x984, not sure why.
master_wav = np.zeros((126936, 90), int)
x = np.linspace(0,5, sample_size)
fs, data = wavfile.read('C:/wav_output/582_hw4_conversion_folder/2_3/Reachy Prints-001-Plaid-OH.wav')


# A random index to start a 5 second sample from
i = random.choice(range(len(data[0:-sample_size])))
sample = data[i:i + sample_size]
frequencies, times, spectrogram = signal.spectrogram(sample, fs)
col = np.reshape(spectrogram,-1)
master_wav[:, 1] = col
test = master_wav[:,1]
#%%
data = data[10*sample_size:11*sample_size]
frequencies, times, spectrogram = signal.spectrogram(data, fs)

plt.pcolormesh(times, frequencies,10*np.log10(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.ylim(-1000, 1000)
plt.show()


#%%
# Iterate through the root folder across all items.
conversion = {'1_1': 'gza', '1_2': 'metallica', '1_3': 'squarepusher', '2_1': 'squarepusher', '2_2': 'aphex', '2_3':'plaid', '3_1':'btbam', '3_2': 'opeth', '3_3': 'tess'}
def get_samples():
    sample_size = 220500
    # Laptop
    rootdir = 'C:/wav_output/582_hw4_conversion_folder'

    # Home Desktop
    # rootdir = 'C:/582_hw4_conversion_folder/wav_output/582_hw4_conversion_folder'
    master_wav = np.zeros((98728, 90), int)
    index = 0
    sample_order = []
    song_order = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # Saves the current folder in order to confirm order of samples
            sample_order.append(os.path.basename(os.path.normpath(subdir)))
            song_order.append(os.path.basename(os.path.normpath(file)))
            # So this does go through the folders in ascending numerical order as we want.
            # Code will look something like:
            file_path = os.path.join(subdir, file)
            fs, data = wavfile.read(file_path)

            # A random index to start a 5 second sample from. Can also return a number of indexes
            # to create multiple samples from each song.
            i = random.choice( range( len( data[2*sample_size:-2*sample_size] ) ) )
            sample = data[i:i+sample_size]
            frequencies, times, spectrogram = signal.spectrogram(sample, fs)
            col = np.reshape(spectrogram, -1)
            # Removes the top 5,000 hz
            master_wav[:, index] = col[0:98728]
            index += 1
    return master_wav, sample_order;
#%% Get 900 at once
# Iterate through the root folder across all items.
conversion = {'1_1': 'gza', '1_2': 'metallica', '1_3': 'squarepusher', '2_1': 'squarepusher', '2_2': 'aphex', '2_3':'plaid', '3_1':'btbam', '3_2': 'opeth', '3_3': 'tess'}
def get_900samples():
    sample_size = 220500
    # Laptop
    rootdir = 'C:/wav_output/582_hw4_conversion_folder'

    # Home Desktop
    # rootdir = 'C:/582_hw4_conversion_folder/wav_output/582_hw4_conversion_folder'
    master_wav = np.zeros((98728, 900), int)
    index = 0
    sample_order = []
    song_order = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # So this does go through the folders in ascending numerical order as we want.
            # Code will look something like:
            file_path = os.path.join(subdir, file)
            fs, data = wavfile.read(file_path)

            # A random index to start a 5 second sample from.
            ten_samples = np.zeros((98728, 10), int)
            for k in np.linspace(0, 9, 10, dtype = int):
                # Saves the current folder in order to confirm order of samples
                sample_order.append(os.path.basename(os.path.normpath(subdir)))
                i = random.choice( range( len( data[2*sample_size:-2*sample_size] ) ) )
                sample = data[i:i+sample_size]
                frequencies, times, spectrogram = signal.spectrogram(sample, fs)
                col = np.reshape(spectrogram, -1)
                ten_samples[:,k] = col[0:98728]
            # Removes the top 5,000 hz
            master_wav[:, index*10:index*10+10] = ten_samples
            index += 1
    return master_wav, sample_order;
#%% Make a large set of 100 samples from each artist
master_data = np.zeros((98728, 900), int)
labels = []
for i in np.linspace(0,9,10, dtype = int):
    A, songs = get_samples()
    master_data[:, i*90:i*90 + 90] = A
    labels += songs
master_labels = [conversion[q] for q in labels]



#%%
# Perform a reduced SVD of the data for Part 1 and plot the singular values on a standard and semi-log axis.
A, sample_order = get_samples()
A1 = A[0:,30:60]

U, S, V = np.linalg.svd(A1, full_matrices=False)
x = np.linspace(1, 30, 30)
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Relative Comparison of Singular Values')
ax1.plot(x, S, 'r-o')
ax1.set_xlabel('$\sigma_j$')
ax1.set_ylabel('$\sigma$ value')

ax2.semilogy(x, S, 'k-o', )
plt.rc('text', usetex=True)
ax2.set_xlabel('$\sigma_j$')
ax2.set_ylabel('log of $\sigma$ value')
plt.show()

#%%
# Plot projection onto dominant POD to look for discrimination
subtitles = ['gza', 'gza', 'gza', 'metallica', 'metallica', 'metallica','squarepusher', 'squarepusher','squarepusher']
plt.figure(1)
fig1, axs1 = plt.subplots(3,3)
x = np.linspace(0, 9, 10)
ymin = -0.7
ymax = 0.7
for i, ax in enumerate(np.reshape(axs1.T, -1)):
    idx = i//3
    ax.plot(x, V[idx*10:idx*10+10, i - 3*idx])
    ax.set_title(subtitles[i])
    ax.set(xlim = (0,10), ylim = (ymin, ymax))
plt.show()
#%% Produce the data matrix
master_data, folders  = get_900samples()
master_labels = [conversion[q] for q in folders]


#%% Test with sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = master_data[:, 0:300].T
classes = master_labels[0:300]

# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Perform the LDA with one component
lda = LDA(n_components=1)
X_train = lda.fit(X_train, y_train)
result = lda.score(X_test,y_test)
print('Score: ' + str(result))
# X_test = lda.transform(X_test)

#%% Test with SVM
from sklearn import svm
# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, random_state=0)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#%% Try SVD first?
data = master_data[:, 300:600]
classes = master_labels[300:600]
U, S, V = np.linalg.svd(data, full_matrices= False)
#%%
# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(V, classes, test_size=0.2, random_state=0)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Perform the LDA with one component
lda = LDA(n_components=1)
X_train = lda.fit(X_train, y_train)
result = lda.score(X_test,y_test)
print('Score: ' + str(result))
# X_test = lda.transform(X_test)

    #%%
# Conventional plotting code
# plt.figure(2)
# fig, axs = plt.subplots(3,3)
# axs[0,0].plot(x, V[0:10, 0])
# axs[0,0].set_title(subtitles[0])
# axs[0,0].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[1,0].plot(x, V[0:10, 1])
# axs[1,0].set_title(subtitles[1])
# axs[1,0].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[2,0].plot(x, V[0:10, 2])
# axs[2,0].set_title(subtitles[2])
# axs[2,0].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[0,1].plot(x, V[10:20, 0])
# axs[0,1].set_title(subtitles[3])
# axs[0,1].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[1,1].plot(x, V[10:20, 1])
# axs[1,1].set_title(subtitles[4])
# axs[1,1].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[2,1].plot(x, V[10:20, 2])
# axs[2,1].set_title(subtitles[5])
# axs[2,1].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[0,2].plot(x, V[20:30, 0])
# axs[0,2].set_title(subtitles[6])
# axs[0,2].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[1,2].plot(x, V[20:30, 1])
# axs[1,2].set_title(subtitles[7])
# axs[1,2].set(xlim=(0, 10), ylim=(ymin, ymax))
# axs[2,2].plot(x, V[20:30, 2])
# axs[2,2].set_title(subtitles[8])
# axs[2,2].set(xlim=(0, 10), ylim=(ymin, ymax))
# plt.show()

#%% LDA
#
# SV_mat = S@V.T
# gza = SV_mat[0:20,0:10]
# metallica = SV_mat[0:20,10:20]
# squarepusher = SV_mat[0:20, 20:30]
#
# m_gza = np.mean(gza, axis = 1)
# m_metall = np.mean(metallica, axis = 1)
# m_square = np.mean(squarepusher, axis = 1)
#
# # within class variance of GZA and Metallica
# SW_gza_metall = 0
# for i in np.linspace(0, 9, 10):
#     SW_gza_metall +=
#%% Testing PD
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)