#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:02:18 2020

@author: pengning
"""

import numpy as np
import scipy as sp

def get_Tvec(ZTT, ZTS_S):
    Tvec = sp.linalg.solve(ZTT, ZTS_S, assume_a='her')
    #Tvec = sp.linalg.solve(ZTT, ZTS_S, assume_a='pos')
    return Tvec


def get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    
    ZTT_chofac = sp.linalg.cho_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    
    Tvec = sp.linalg.cho_solve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(sp.linalg.cho_solve(ZTT_chofac, -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return Tvec, gradTvec
