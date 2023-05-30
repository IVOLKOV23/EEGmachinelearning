import numpy as np
import os
import csv
from collections import defaultdict
import subprocess as sp
import psutil
import sys
import matplotlib.pyplot as plt
import scikitplot as skplt
import cv2 as cv
import json
import re


def save_dict(filename, dict):
    with open(filename, 'w') as fp:
        # encoder/decoder
        # convert into json object
        json.dump(dict, fp)


def load_dict(filename):
    with open(filename) as fp:
        # loading a dictionary from json
        dict = json.load(fp)

    return dict


class mydict(defaultdict):
    # allows dict to be reference by adding .<key> to the variable.
    # if you use .<key> and it hasn't been created will return []
    # This is useful for collating data

    __getattr__ = dict.__getitem__

    def __init__(self, *args, **kwargs):
        # this is where the default type is declared. if you want something else change it here.
        super().__init__(list, *args,
                         **kwargs)

    def isempty(self):

        # check if there is anything in self
        if len(self.keys) == 0:
            return True

        else:
            return False

    def as_dict(self):
        # map a dict from self
        return dict(self)

    def appends(self, d: dict):
        # appends all items in this dictionary to the same keys in this dictionary.
        # this is useful when collecting data for a number of different keys
        for key, value in d.items():
            self[key].append(value)

    def means(self):
        # calculates the means of each element in dict. Make sure that every key is a list of values.
        # Replaces the values with its mean
        for key, value in self.items():
            if len(value) > 0:
                # try for errors
                try:
                    self[key] = np.mean(value)
                except:
                    print(f" line 192. key failed")
                    pass
            pass
        return self

    def save(self, epoch, folder):
        # appends data to csv file as {key}.csv
        for key, value in self.items():
            value = np.array(value)
            # sz=value.size
            if key == 'shape' or value.size == 0: continue
            file = os.path.join(folder, f"{key}.csv")
            with open(file, mode='a', newline='') as employee_file:
                writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, value.tolist()])


def chunks(lst, n, trim=True):
    """Yield successive n-sized chunks from lst.
    trim==True means that each chunk is exactly the same size discarding the last portion of data.
    """

    size = len(lst)
    for i in range(0, size, n):
        if trim and (i + n) > size:  # TODO: MAYBE SHOULD BE >=
            break
        else:
            yield lst[i:i + n]


def get_performance_stats(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, F1, precision, recall


def get_file_nums(filename):
    ##extracts a list of integers from a string. good for getting numbers from a filename when automating saving
    regex = re.compile(r'\d+')
    res = [int(x) for x in regex.findall(filename)]

    return res


def gpu_memory_usage(gpu_id, output=True):
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    mem_cmd = sp.check_output(command.split())
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.free --format=csv"
    util_cmd = sp.check_output(command.split())

    memory_used = mem_cmd.decode("ascii").split("\n")[1]
    mem_free = util_cmd.decode("ascii").split("\n")[1]
    # perc_used = perc_used[:-2]#perc_used.decode("ascii").split("%")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0]) * 1000000
    memory_free = int(mem_free.split()[0]) * 1000000

    if output: print(f"GPU: {humanbytes(memory_used)}, {round(100 * memory_used / (memory_used + memory_free), 2)}% ")

    return memory_used, memory_free


def getmem(output=True):
    result = psutil.virtual_memory()
    btes = result[3]
    perc = result[2]
    if output: print(f"RAM: {humanbytes(btes)}, {perc}% ")
    return btes, perc


def mem(gpu=0, output=True):
    btes, perc = getmem(output=False)
    memory_used, memory_free = gpu_memory_usage(gpu, output=False)
    if output: print(
        f"RAM: {humanbytes(btes)}, {perc}% - GPU: {humanbytes(memory_used)}, {round(100 * memory_used / (memory_used + memory_free), 2)}% ")
    return btes, perc, memory_used, memory_free


def report_mem(local_vars):
    bytes, perc, memory_used, memory_free = mem()
    print("**********************MEM USAGE******************")
    variables = []
    for var, obj in local_vars:
        variables.append([sys.getsizeof(obj), var])
    count = 0
    variables.sort(reverse=True)
    for size, var in variables:
        if count > 10:
            break
        print(var, humanbytes(size))
        count += 1


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'B' if 0 == B > 1 else 'B')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def check_usage(perc):
    if perc > 90:
        print("**********************LARGE MEM USAGE******************")
        local_vars = list(locals().items())
        variables = []
        for var, obj in local_vars:
            variables.append([sys.getsizeof(obj), var])
        count = 0
        variables.sort(reverse=True)
        for size, var in variables:
            if count > 10:
                break
            print(var, humanbytes(size))
            count += 1
