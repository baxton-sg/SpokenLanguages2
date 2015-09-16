

import os
import numpy as np
from utils import *
from ann_rbm import ANN


MIN = -32768
RAN = 65536

FLT_NUM = 3


def get_files():
    files = [f for f in os.listdir(path) if "wav" in f]
    return files




def process(files):
    F = 4
    S = 1
    ss = [F, 1, F]

    anns = []

    for f in range(FLT_NUM):
        anns.append( ANN(ss, 0.) )
        ww, bb = anns[-1].get_weights()
        ww = np.random.rand(ww.shape[0])
        bb = np.random.rand(bb.shape[0])
        anns[-1].set_weights(ww, bb)


    total_files = [0.] * FLT_NUM
    total_cost = [0.] * FLT_NUM

    cnt = [0.] * FLT_NUM
    cost = [0.] * FLT_NUM

    for e in range(10000):
        for f in files:
            data = read(path + "\\" + f)
 
            data = (data - MIN) / RAN
             
            W = data.shape[0]
            for b in range(0, W - F, S):
                tmp = data[b:b+F]

                for f in range(FLT_NUM):
                    anns[f].partial_fit(tmp, tmp)

                    cost[f] += anns[f].cost.value
                    total_cost[f] += anns[f].cost.value

                    total_files[f] += 1.
                    cnt[f] += 1.

                if cnt[0] == 10000.:
                    for f in range(FLT_NUM):
                        print "ann[", f, "] ====="
                        print "    files", total_files[f], "avr cost", (cost[f] / cnt[f]), "total avr cost", (total_cost[f] / total_files[f])
                        cnt[f] = 0.
                        cost[f] = 0.












def main():
    np.random.seed()

    files = get_files()
    process(files)




main()
