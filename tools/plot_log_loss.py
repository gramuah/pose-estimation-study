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
    key2 = sys.argv[2]
    
    print "Processing log file: ", log_file
    
    loss_v = []
    loss2_v = []
    it_v = []
    with open(log_file,'r') as f:
        for line in f:
            s = line.split(' ')
            if key in line:
#                print s[-1]
		loss_v.append( float(s[-1]) )
            if 'solver.cpp:218' in line:
#		print s[5]
                it_v.append( int( s[6]) ) 
            if key2 in line:
                loss2_v.append( float(s[-2]) )

    # Median filter
    med_loss = sig.medfilt(loss_v, 101)
    med_loss2 = sig.medfilt(loss2_v, 101)

#    plt.plot(it_v, loss_v, 'b')
#    plt.plot(it_v, med_loss, 'r', lw=2.0)
    plt.plot(it_v, loss2_v, 'g')
    plt.plot(it_v, med_loss2, 'y', lw=2.0)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
