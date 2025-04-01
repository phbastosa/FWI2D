import numpy as np

def catch_parameter(parameters, target):
    file = open(parameters, "r")
    for line in file.readlines():
        if line[0] != "#":
            splitted = line.split()
            if len(splitted) != 0:
                if splitted[0] == target: 
                    return splitted[2]     

def read_binary_array(n1,filename):
    return np.fromfile(filename, dtype = np.float32, count = n1)    

def read_binary_matrix(n1,n2,filename):
    data = np.fromfile(filename, dtype = np.float32, count = n1*n2)   
    return np.reshape(data, [n1, n2], order='F')