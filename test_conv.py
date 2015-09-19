import os
import numpy as np
from conv import CONV



FOR_UNIX = True

SEP = "\\" if not FOR_UNIX else "/"

path_base = "C:\\Temp\\test_python\\NLR\\data\\" if not FOR_UNIX else "/home/maxim/kaggle/NLP/data/"
path_in   = path_base + "train_mp3" + SEP
#path_in   = path_base + "wavs" + SEP



MIN = -32768
RAN = 65536

MIN = 0
RAN = 2 ** 32

print MIN, RAN

MIN_COST = 0.0009

FLT_NUM = 2

F = 16
S = 2


def get_files():
    files = [f for f in os.listdir(path_in)]
    return files





def process(files):

    conv = CONV(FLT_NUM, F=F, S=S, min_cost=MIN_COST)

    stop = False
    cnt = 0

    for e in range(10000):
        if stop:
            break

        for f in files:
            #data = np.fromfile(path_in + SEP + f, dtype=np.int16, sep='').astype(np.float64)
            data = np.fromfile(path_in + SEP + f, dtype=np.uint32, sep='').astype(np.float64)

            width = data.shape[0]
            width = width if 0 == (width % 2) else width - 1
            data = data[:width].reshape((width/2,2)).max(axis=1)
    
            data = (data - MIN) / RAN

            stop = conv.fit(data)

            cnt += 1
            if 0 == (cnt % 100):
                print "Epoch", e, "files", cnt
                costs = conv.avr_cost()
                for c in range(len(costs)):
                    print "    [%d]" % c, costs[c]
            
            if stop:
                break









def main():
    files = get_files()
    process(files)




main()
