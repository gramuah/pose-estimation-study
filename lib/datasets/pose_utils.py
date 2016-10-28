# -*- coding: utf-8 -*-
'''
    @author: Daniel Oñoro Rubio
    @organization: GRAM, University of Alcalá, Alcalá de Henares, Spain
    @copyright: See LICENSE.txt
    @contact: daniel.onoro@edu.uah.es
    @date: 27/10/2016
'''

import numpy as np

def generate_interval(num_bins):
    """
    Return an array with the limist of a 'num_bins' binarization of 360 degrees.
    """
    interval = np.arange(360.0/(2*num_bins),360.0-360.0/(2*num_bins)+1,360.0/num_bins)
    return interval

def find_interval(azimuth, interval_v):
    """
    Compute the index of a binarized angle
    """
    if azimuth > interval_v[-1]:
        return 0
    else:          
        return np.where( azimuth <= interval_v )[0][0]

if __name__ == '__main__':
    pass