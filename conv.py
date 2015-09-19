


import numpy as np
from ann import ANN





class CONV(object):
    def __init__(self, ann_num, F=16, S=2, verbose=True, min_cost=.0007):
        self.F = F
        self.S = S

        self.W = 0
        self.H = 0
        self.D = 0

        self.min_cost = min_cost

        self.initialized = False

        self.ss = None
        self.anns = []
        self.FLT_NUM = ann_num

        self.verbose = verbose
        self.total_vectors = 0.
        self.total_cost = [0.] * self.FLT_NUM

        self.stop = [False] * self.FLT_NUM
    
        np.random.seed()



    def twick_F(self, width):
        fr = float(width - self.F) / self.S
        while fr != int(fr):
            self.F += 1
            fr = float(width - self.F) / self.S


    # WARN: this is implementation for 1D array or 1D + depth
    #       for 2D or 2D + depth this should be modified
    def fit(self, data):
        if not self.initialized:
            if len(data.shape) == 1:
                self.W = data.shape[0]
                self.D = 1
                self.H = 1

            elif len(data.shape) == 2:
                self.W = data.shape[0]
                self.D = data.shape[1]
                self.H =1 

            #elif len(data.shape) == 3:
            else:
                raise Exception("input shape of size %d is not supported" % len(data.shape))

            self.twick_F(self.W)

            input_size = self.D * self.F
            self.ss = [input_size, 1, input_size]
            for f in range(self.FLT_NUM):
                self.anns.append( ANN(self.ss, .0) )

                # re initialize weights
                ww, bb = self.anns[-1].get_weights()
                ww = (np.random.rand(ww.shape[0]) - .5)
                bb = (np.random.rand(bb.shape[0]) - .5)
                self.anns[-1].set_weights(ww, bb)

            self.initialized = True    

            if self.verbose:
                print "CNN: num", self.FLT_NUM, ", F", self.F, ", S", self.S, ", WIDTH", self.W, ", D", self.D, ", input size", input_size
        else:
            # sanity check
            if self.W != data.shape[0]:
                print "WARN: expected width", self.W, "provided", data.shape[0]
                return False
        


        # TODO this is 1D implementation !!!

        #width = self.F * self.D
        #step  = self.S * self.D
        #N     = self.W / self.D - width

        if True:
        #for b in range(0, N, self.S):
        #    tmp = data[b : b + width]

            #if self.verbose:
            #    print "beg", beg, "end", end, "size", tmp.shape, ",", tmp

            self.total_vectors += 1.
            for f in range(self.FLT_NUM):
                if self.stop[f]:
                    continue
                self.anns[f].conv_fit(data, W=self.W, F=self.F, S=self.S, D=self.D)
                self.total_cost[f] += self.anns[f].cost.value


        for f in range(self.FLT_NUM):
            if (self.total_cost[f] / self.total_vectors) < self.min_cost:
                self.stop[f] = True

        return self.finished()
           
 

    def finished(self):
        return self.FLT_NUM == np.sum(self.stop)



    def avr_cost(self):
        costs = []
        for f in range(self.FLT_NUM):
            costs.append( self.total_cost[f] / self.total_vectors )
        return costs








