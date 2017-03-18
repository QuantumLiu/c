# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 07:48:21 2017

@author: Quantum Liu
"""
from multiprocessing import Pool
from time import sleep
import h5py
import scipy.io as sci
import numpy as np

def convert_array(line):
    ID,label,radar_map=line.split('\n')[0].split(',')
    data = radar_map.split(' ')
    data = np.array(data, dtype=np.int8)
    data = np.reshape(data, (15,4,101,101))
    return [data,label,ID]
def main():
    f=open('train.txt','r')
    pool = Pool(processes=56)    # set the processes max number 3
    results=[]
    for line in f.readlines():
        result=pool.apply_async(convert_array, (line,))
        results.append(result)
    pool.close()
    pool.join()
    data=[]
    label=[]
    ID=[]
    for result in results:
        a=result.get()
        data.append(a[0])
        label.append(a[1])
        ID.append(a[2])
    data=np.array(data)
    label=np.array(label, dtype=np.float)
    ID=np.array(ID)
    file=h5py.File('train/dataset_train.h5','w')
    file.create_dataset('data', data = data)
    file.create_dataset('label', data = label)
    sci.savemat('ID.mat',{'ID':ID})    
    file.close()
    f.close()
if __name__ == "__main__":
    main()