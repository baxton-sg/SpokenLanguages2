

import numpy as np
from scipy.io import wavfile as wf
from ann_rbm import ANN




path = r'C:\Temp\test_python\NLR\data\wavs'


stream = np.fromfile(path + "\\000kouqjfnk.mp3.wav", dtype=np.int16, sep='')

header = stream[:23]

stream = stream[23:]
stream = stream.reshape((stream.shape[0]/2,2))



print stream.shape
print stream


###w, stream = wf.read(path + "\\000kouqjfnk.mp3.wav")
s1 = stream[:,0].astype(np.float64)
s2 = stream[:,1].astype(np.float64)

s1 = s1.reshape((s1.shape[0], 1))
s2 = s2.reshape((s2.shape[0], 1))


print s1
print s2




# remove one channel and save it back to the disk
with open(path + "\\..\\test.wav", "wb+") as fout:
    header.tofile(fout, sep='')
    s = np.concatenate((s1, s1), axis=1).astype(np.int16)
    print s
    s.tofile(fout, sep='')
