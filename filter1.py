

import os
import numpy as np
from ann_rbm import ANN


FOR_UNIX = False


SEP = "\\" if not FOR_UNIX else "/"


path_base = "C:\\Temp\\test_python\\NLR\\data\\" if not FOR_UNIX else ""
path_in   = path_base + "wavs" + SEP
path_out  = path_base + "flt1" + SEP

MIN = -32768
RAN = 65536

MIN_COST = 0.0009

FLT_NUM = 3

F = 16
S = 2


def get_files():
    files = [f for f in os.listdir(path_in)]
    return files





def save_ann(fout, ann, ann_num):
    ww, bb = ann.get_weights()

    fout.write("ww_size%d = %d\n" % (ann_num, ww.shape[0]))
    fout.write("ww%d = np.array([" % ann_num)
    for w in ww:
        fout.write("%f," % w)
    fout.write("], dtype=np.float64)\n\n")
    
    fout.write("bb_size%d = %d\n" % (ann_num, bb.shape[0]))
    fout.write("bb%d = np.array([" % ann_num)
    for b in bb:
        fout.write("%f," % b)
    fout.write("], dtype=np.float64)\n\n")
    
    fout.write("ss%d = [F, 1]\n" % ann_num)
    fout.write("anns.append( ANN(ss%d, .0) )\n" % ann_num)
    fout.write("anns[-1].set_weights(ww%d[:ww_size%d / 2], bb%d[:1])\n" % (ann_num, ann_num, ann_num))
  
    fout.write("\n\n")

    





def save(fout, anns):
    N = len(anns)
    fout.write("import numpy as np\n")
    fout.write("from ann import ANN\n")
    fout.write("\n\n")
    fout.write("# Filters [%s]\n" % (N))
    fout.write("MIN = -32768\n")
    fout.write("RAN = 65536\n")
    fout.write("FLT_NUM = %d\n" % FLT_NUM)
    fout.write("F = %d\n" % F)
    fout.write("S = %d\n" % S)
    fout.write("\n\n")

    fout.write("anns = []\n")
    fout.write("\n\n")


    for a in range(N):
        save_ann(fout, anns[a], a)





def process(files):
    ss = [F, 1, F]

    anns = []

    # looks like my seed in C++ doesnt work, so re-assign weights here
    for f in range(FLT_NUM):
        anns.append( ANN(ss, 0.) )
        ww, bb = anns[-1].get_weights()
        ww = np.random.rand(ww.shape[0])
        bb = np.random.rand(bb.shape[0])
        anns[-1].set_weights(ww, bb)



    stop = [False] * FLT_NUM


    total_files = [0.] * FLT_NUM
    total_cost = [0.] * FLT_NUM

    cnt = [0.] * FLT_NUM
    cost = [0.] * FLT_NUM

    for e in range(10000):
        for f in files:
            data = np.fromfile(path_in + SEP + f, dtype=np.int16, sep='').astype(np.float64)
 
            data = (data - MIN) / RAN
             
            W = data.shape[0]
            for b in range(0, W - F, S):
                tmp = data[b:b+F]

                for f in range(FLT_NUM):
                    if stop[f]:
                        continue

                    anns[f].partial_fit(tmp, tmp)

                    cost[f] += anns[f].cost.value
                    total_cost[f] += anns[f].cost.value

                    total_files[f] += 1.
                    cnt[f] += 1.

                    if (total_cost[f] / total_files[f]) < MIN_COST:
                        stop[f] = True

                if cnt[0] == 10000.:
                    for f in range(FLT_NUM):
                        print "ann[", f, "] ====="
                        print "    files", total_files[f], "avr cost", ((cost[f] / cnt[f]) if cnt[f] != 0 else 0), "total avr cost", (total_cost[f] / total_files[f])
                        cnt[f] = 0.
                        cost[f] = 0.




                if np.sum(stop) == FLT_NUM:
                    break

            if np.sum(stop) == FLT_NUM:
                break

        if np.sum(stop) == FLT_NUM:
            break


    with open("flt_anns_1.py", "w+") as fout:
        save(fout, anns)

    return anns








def transform(files, anns):
    for f in files:
        data = np.fromfile(path_in + SEP + f, dtype=np.int16, sep='').astype(np.float64)
        data = (data - MIN) / RAN

        row = np.zeros((FLT_NUM,), dtype=np.float32)
             
        W = data.shape[0]

        with open(path_out + f, "wb+") as fout:
            for b in range(0, W - F, S):
                tmp = data[b:b+F]
    
                for f in range(FLT_NUM):
                    p = anns[f].predict_proba(tmp)[0,1]
                    row[f] = p

                row.tofile(fout, sep='')







def main():
    np.random.seed()

    if not os.path.exists(path_out):
        raise Exception("out path doesnot exist [" + path_out + "]")

    files = get_files()
    anns = process(files)

    print "Start transform"

    transform(files, anns)

    print "DONE"




main()
