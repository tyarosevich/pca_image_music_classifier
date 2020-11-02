import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
import random

    #%% #%% Defines a function to get samples from my folder with processed .wav files

# This dictionary gives the folder names as keys to the artists inside them.
conversion = {'1_1': 'gza', '1_2': 'metallica', '1_3': 'squarepusher', '2_1': 'squarepusher', '2_2': 'aphex', '2_3':'plaid', '3_1':'btbam', '3_2': 'opeth', '3_3': 'tess'}

# A sample that returns 900 samples and the associated artist folder name.
def get_900samples():
    sample_size = 220500
    # Laptop
    rootdir = 'C:/wav_output/582_hw4_conversion_folder'

    # Home Desktop
    # rootdir = 'C:/582_hw4_conversion_folder/wav_output/582_hw4_conversion_folder'

    master_wav = np.zeros((35000, 900), int)
    index = 0
    sample_order = []
    song_order = []

    # Iterates through the base folder with os.walk, which goes through each subfolder, with
    # iterate variables for the files, the subdirectories and the directories.
    for subdir, dirs, files in os.walk(rootdir):

        # Iterates through each file.
        for file in files:
            # Each filepath, should we want to save it for any reason.
            file_path = os.path.join(subdir, file)
            # Reads in the .wav file
            fs, data = wavfile.read(file_path)
            # A holder for the ten samples we take from each song.
            ten_samples = np.zeros((35000, 10), int)
            # An iteration to take the 10 samples.
            for k in np.linspace(0, 9, 10, dtype = int):
                # Saves the current folder in order to confirm order of samples
                sample_order.append(os.path.basename(os.path.normpath(subdir)))
                # A random index to start the sample from.
                i = random.choice( range( len( data[2*sample_size:-2*sample_size] ) ) )
                sample = data[i:i+sample_size]
                # Takes a spectrogram of the file and reshapes it into a vector.
                frequencies, times, spectrogram = signal.spectrogram(sample, fs)
                col = np.reshape(spectrogram, -1)
                ten_samples[:,k] = col[0:35000]
            # Removes the top 5,000 hz
            master_wav[:, index*10:index*10+10] = ten_samples
            index += 1
    return master_wav, sample_order;

#%% Produce the data matrix
master_data, folders  = get_900samples()
# Returns a list of values associated with a list of keys.
master_labels = [conversion[q] for q in folders]

#%%
# Perform a reduced SVD of the data for Part 1 and plot the singular values on a standard and semi-log axis.
# Variables to change to re-run the code for parts 1, 2, 3.
start = 600
stop = 900
A1 = master_data[0:,start:stop]

U, S, V = np.linalg.svd(A1, full_matrices=False)
x = np.linspace(1, 50, 50)

# Plots the first 50 singular values.
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Relative Comparison of Singular Values')
ax1.plot(x, S[0:50], 'r-o')
ax1.set_xlabel('$\sigma_j$')
ax1.set_ylabel('$\sigma$ value')

ax2.semilogy(x, S[0:50], 'k-o', )
plt.rc('text', usetex=True)
ax2.set_xlabel('$\sigma_j$')
ax2.set_ylabel('log of $\sigma$ value')
plt.show()

#%%
# Plot projection onto dominant POD to look for discrimination
# subtitles = ['gza', 'gza', 'gza', 'metallica', 'metallica', 'metallica','squarepusher', 'squarepusher','squarepusher']
# subtitles = ['squarepusher', 'squarepusher','squarepusher', 'aphex twin', 'aphex twin', 'aphex twin', 'plaid', 'plaid', 'plaid']
subtitles = ['thrash', 'thrash', 'thrash', 'prog', 'prog', 'prog', 'death', 'death', 'death']
plt.figure(1)
fig1, axs1 = plt.subplots(3,3)
x = np.linspace(0, 5, 10)
ymin = -0.1
ymax = 0.1
# A highly compacted iteration to plot 9 subplots.
for i, ax in enumerate(np.reshape(axs1.T, -1)):
    idx = i//3
    ax.plot(x, V[i - 3*idx , idx*100:idx*100+10])
    ax.set_title(subtitles[i])
    ax.set(xlim = (0,5), ylim = (ymin, ymax))
    ax.set_xlabel('time(s)')
    ax.set_ylabel('POD value')
plt.show()


#%% Test with sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = master_data[:, start:stop].T
classes = master_labels[start:stop]

# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Perform the LDA with one component
lda = LDA(n_components=2)
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

clf = svm.SVC(kernel)
clf.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#%% Test with QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
2
clf = QDA()
clf.fit(X_train, y_train)
print('Accuracy of QDA classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of QDA classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#%% Test with Naive Bayes
from sklearn.naive_bayes import GaussianNB as GNB
# Splits the data into a training set and randomized test set with accompanying labels
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2)

# Scales the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = GNB()
clf.fit(X_train, y_train)
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
#%% KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
