#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:49:47 2018

@author: tina
Adapted from the course Computational Intelligence at TU Graz
"""

import matplotlib.pyplot as plt

def plot_score_vs_C(train_scores, test_scores, C_list):
    """
    Plot the score as a function of the number of polynomial degree.
    :param train_scores: List of training scores, one for each polynomial degree
    :param test_scores: List of testing scores, one for each polynomial degree
    :param poly_degree_list: List containing degrees of the polynomials corresponding to each of the scores.
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training scores with different values of C")
    plt.plot(C_list, train_scores, 'o', linestyle='-', label="Training scores", lw=2)
    plt.plot(C_list, test_scores, 'o', linestyle='-', label="Testing scores", lw=2)
    plt.xlabel("C")
    plt.ylabel("Score (mean accuracy)")
    plt.legend()
    plt.show()

def plot_score_vs_degree(train_scores, test_scores, poly_degree_list):
    """
    Plot the score as a function of the number of polynomial degree.
    :param train_scores: List of training scores, one for each polynomial degree
    :param test_scores: List of testing scores, one for each polynomial degree
    :param poly_degree_list: List containing degrees of the polynomials corresponding to each of the scores.
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training scores with polynomial degrees")
    plt.plot(poly_degree_list, train_scores, 'o', linestyle='-', label="Training scores", lw=2)
    plt.plot(poly_degree_list, test_scores, 'o', linestyle='-', label="Testing scores", lw=2)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Score (mean accuracy)")
    plt.legend()
    plt.show()
    
def plot_score_vs_gamma(train_scores, test_scores, gamma_list):
    """
    Plot the score as a function of the number of polynomial degree.
    :param train_scores: List of training scores, one for each gamma
    :param test_scores: List of testing scores, one for each gamma
    :param gamma_list: List containing gammas corresponding to each of the scores.
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training scores with gamma")
    plt.plot(gamma_list, train_scores, linestyle='-', label="Training scores", color='blue', lw=2)
    plt.plot(gamma_list, test_scores, linestyle='-', label="Testing scores", color='green', lw=2)

   
    plt.xlabel("Value of \gamma")
    plt.ylabel("Score (mean accuracy)")
    plt.legend()
    plt.show()