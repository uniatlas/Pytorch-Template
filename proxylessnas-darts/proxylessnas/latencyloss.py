'''
@Description: Latency Loss implement
@Author: xieydd
@Date: 2019-09-05 10:26:56
@LastEditTime: 2019-09-06 16:28:06
@LastEditors: Please set LastEditors
'''
import csv
import proxylessnas

import torch
import torch.nn as nn
from genotypes import PRIMITIVES


class LatencyLoss(nn.Module):
    def __init__(self, channels, steps, strides, input_size=56):
        super(LatencyLoss, self).__init__()

        self.channels = channels
        self.steps = steps
        self.strides = strides

        self._calculate_feature_map_size(input_size)
        self._load_latency()

    def _load_latency(self):
        # load predicted latency file
        with open('../proxylessnas/latency.csv') as f:
            rdr = csv.reader(f)

            self._latency = {}
            for line in rdr:
                self._latency[line[0]] = line[1]
        f.close()

    def _calculate_feature_map_size(self, input_size):
        self.feature_maps = [input_size]
        for s in self.strides[:-1]:
            input_size = input_size // s
            self.feature_maps.append(input_size)

    def _predictor(self, inputs):
        """predict latency
        input example: mbconv_6_3_80_80_14_1
        """
        div = inputs.split('_', maxsplit=-1)
        if div[0] == 'identity' or div[0] == 'none':
            div.insert(1, 0)  # insert fake exp_rate
            div.insert(2, 0)  # insert fake ksize
        op, exp_rate, ksize, C_in, C_out, size, stride = div
        #print(op)
        if op == 'identity' or op == 'none':
            return 0
        out_size = int(size) // int(stride)
        findstr = '{}x{}x{}-{}x{}x{}-expand:{}-kernel:{}-stride:{}'.format(size,size,C_in,out_size,out_size,C_out,exp_rate,ksize,ksize) 
        #print(findstr)
        return float(self._latency.get(findstr))

    def forward(self, alpha):
        latency = 0

        for i, a_cell in enumerate(alpha):
            c_in = self.channels[i]
            c_out = self.channels[i+1]
            fm = self.feature_maps[i]
            strides = self.strides[i]

            for j, weights in enumerate(a_cell):
                op_names = PRIMITIVES
                #strides = 1 if j != 0 else strides
                latency += sum(w * self._predictor('{}_{}_{}_{}_{}'.format(op,c_in,c_out,fm,strides)) for w, op in zip(weights, op_names))
        return latency
