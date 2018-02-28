#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:29:24 2018

@author: tina
Adapted from the course Computational Intelligence at TU Graz
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import warnings

from data_an import uredi_podatke
from svm_plot import plot_score_vs_gamma, plot_score_vs_degree, plot_score_vs_C

def main():
    #uread data
    data_train = pd.read_csv('train.csv')
    training_scores = list()
    testing_scores = list()
    x_train, y_train, x_test, y_test = uredi_podatke(data_train)
    scale(x_train, y_train, x_test, y_test)
    
#    for i in range(10):
#        x_train, y_train, x_test, y_test = uredi_podatke(data_train)
#        scale(x_train, y_train, x_test, y_test)
#        
#        #Pocetni rezultati:
#        #train_score, test_score = svm_linear(x_train, y_train, x_test, y_test)
#        train_score, test_score = svm_rbf(x_train, y_train, x_test, y_test)
##        train_score, test_score = svm_poly(x_train, y_train, x_test, y_test)
#        training_scores.append(train_score)
#        testing_scores.append(test_score)
#        
#    train_best = max(training_scores)
#    test_best = max(testing_scores)
#    train_worst = min(training_scores)
#    test_worst = min(testing_scores)    
#    train_avg = np.average(training_scores)
#    test_avg = np.average(testing_scores)
#    print(train_worst, test_worst, train_best, test_best)
#    print("Prosjecni training score =", train_avg, ", prosjecni testing score = ", test_avg)
    
    #svm_linear( x_train, y_train, x_test, y_test)
    #svm_poly( x_train, y_train, x_test, y_test)
    svm_rbf( x_train, y_train, x_test, y_test)
    
def scale(x_train, y_train, x_test, y_test):
    s = MinMaxScaler()    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    x_train = pd.DataFrame(s.fit_transform(x_train), columns=x_train.columns)
    y_train = pd.DataFrame(s.fit_transform(y_train))
    x_test = pd.DataFrame(s.fit_transform(x_test), columns=x_test.columns)
    y_test = pd.DataFrame(s.fit_transform(y_test))
    return x_train, y_train, x_test, y_test
    
def svm_linear(x_train, y_train, x_test, y_test):
    # Linear kernel
#    svc = svm.SVC(kernel='linear')
#    svc.fit(x_train,y_train)
#    train_score = svc.score(x_train, y_train)
#    test_score = svc.score(x_test, y_test)
    #print("Linear kernel: training score =", train_score, "testing score =", test_score)
    #return train_score, test_score
    
    #Cs = [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1 ]
    Cs = np.arange(0.001, 0.1, 0.002)
    training_scores = list()
    test_scores = list()
    for C in Cs:
        svc = svm.SVC(kernel='linear', C=C)
        svc.fit(x_train,y_train)
        training_scores.append(svc.score(x_train, y_train))
        test_scores.append(svc.score(x_test, y_test))
    test_best = test_scores.index(max(test_scores))
    best_C=Cs[test_best]
    plot_score_vs_C(training_scores, test_scores, Cs)
    print("Linear kernel:")
    print("Best C is ", best_C, " with testing score=", max(test_scores))
    linear_best = svm.SVC(C=best_C)
    linear_best.fit(x_train, y_train)
    

def svm_rbf(x_train, y_train, x_test, y_test):    
    # RBF kernel
#    svc = svm.SVC()
#    svc.fit(x_train,y_train)
#    train_score = svc.score(x_train, y_train)
#    test_score = svc.score(x_test, y_test)
#    return train_score, test_score
#
    gammas = np.arange(0.01, 2, 0.02)
    training_scores = list()
    test_scores = list()
    for g in gammas:
        svc = svm.SVC(kernel='rbf', gamma=g)
        svc.fit(x_train, y_train)
        training_scores.append(svc.score(x_train, y_train))
        test_scores.append(svc.score(x_test, y_test))
    test_best = test_scores.index(max(test_scores))
    best_gamma=gammas[test_best]
    plot_score_vs_gamma(training_scores, test_scores, gammas)
    print("RBF kernel:")
    print("Best gamma is ", best_gamma, " with testing score=", max(test_scores))
    rbf_best = svm.SVC(gamma=best_gamma)
    rbf_best.fit(x_train, y_train)
    
 
def svm_poly(x_train, y_train, x_test, y_test):
    #Polynomial kernel
#    svc = svm.SVC(kernel='poly')
#    svc.fit(x_train,y_train)
#    train_score = svc.score(x_train, y_train)
#    test_score = svc.score(x_test, y_test)
#    return train_score, test_score

    degrees = range(1, 8)
    training_scores = list()
    test_scores = list()
    for i in degrees:
        svc = svm.SVC(kernel="poly", degree=i)
        svc.fit(x_train, y_train)
        training_scores.append(svc.score(x_train, y_train))
        test_scores.append(svc.score(x_test, y_test))
    degree_best = test_scores.index(max(test_scores)) + 1
    plot_score_vs_degree(training_scores, test_scores, degrees)
    print("Polynomial kernel:")
    print("Best degree is ", degree_best, " with testing score=", max(test_scores))
    poly_best = svm.SVC(kernel="poly", degree=degree_best, coef0=1)
    poly_best.fit(x_train, y_train)

if __name__ == '__main__':
    main()