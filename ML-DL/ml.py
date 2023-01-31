import time
import numpy as np
 
# manipulating a 10 million items
n = 10000000
 
 
def numpy_add():
    # time in seconds
    now = time.time()
    # generating a 10 million items
    x = np.arange(n)
    # generating a 10 million items
    y = np.arange(n)
    result = x + y
    return time.time() - now
 
 
def python_add():
    # time in seconds
    now = time.time()
    # generating a 10 million items
    x = range(n)
    # generating a 10 million items
    y = range(n)
    result = [x[i] + y[i] for i in range(len(x))]
    return time.time() - now
 
 
time_python = python_add()
time_numpy = numpy_add()
print('Python time: ' + str(time_python) + 's')
print('NumPy time: ' + str(time_numpy) + 's')
print("Numpy is " + str(time_python/time_numpy) + "x faster!")