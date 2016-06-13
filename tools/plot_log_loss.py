#!/usr/bin/env python

'''
Created on Jan 5, 2016

@author: Daniel Onoro Rubio
'''

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys

key = ", loss ="

if __name__ == '__main__':
    log_file = sys.argv[1]
    
    print "Processing log file: ", log_file
    
    loss_v = []
    it_v = []
    with open(log_file,'r') as f:
        for line in f:
            if key in line:
                s = line.split(' ')
                loss_v.append( float(s[-1]) )
                it_v.append( int( s[-4][:-1]) ) 

    # Median filter
    med_loss = sig.medfilt(loss_v, 101)

    plt.plot(it_v, loss_v, 'b')
    plt.plot(it_v, med_loss, 'r', lw=2.0)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()