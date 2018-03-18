#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 02:19:08 2018

@author: caoxiya
"""

from __future__ import division
import zhou.predictDepth as zpD
import laina.predictDepth as lpD
import argparse
import evalmatrix
import shutil
import os
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["zhou", "laina", "liu", "eigen"], help="method to evaluate")
#parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

def prepareData():
    filelist = []
    with open('test.txt') as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        filelist.append('kitti_raw_data/' + line[:-1])
    return filelist

def prepareMatlab(dest):
    filelist = prepareData()
    pwd = os.getcwd()
    i = 0
    for file_name in filelist:
        full_file_name = pwd + '/' + file_name
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest + "/" + str(i) + ".png")
            i = i + 1

    
def calculateDepth(method):
    filelist = prepareData()
    pwd = os.getcwd()
    if method == "zhou":
        depth = zpD.predictDepth(filelist, pwd)
    elif method == "laina":
        depth = lpD.predictDepth(filelist, pwd)
    elif method == "liu":
        prepareMatlab(pwd + "/liu/test")
        print "Run Matlab!"
    else:
        print "Have not implement!"
    return depth

def main():
    depth = calculateDepth(args.method)
    print depth.shape
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evalmatrix.eval(depth, depth)
    print "abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3", abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
        
main()