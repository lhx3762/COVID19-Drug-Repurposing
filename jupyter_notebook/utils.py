import os


def read_dict_file(file):
    dic = ''
    with open(file,'r') as f:
        for line in f.readlines():
            dic = line # string
    dic = eval(dic)
    return(dic)