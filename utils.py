



import numpy as np




path = r'C:\Temp\test_python\NLR\data\wavs'


def read(wav_file_name):
    stream = np.fromfile(wav_file_name, dtype=np.int16, sep='')

    header = stream[:23]

    stream = stream[23:]
    stream = stream.reshape((stream.shape[0]/2,2))

    s1 = stream[:,0].astype(np.float64)


    return s1
