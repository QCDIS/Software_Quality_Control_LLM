#!/usr/bin/env python
# coding: utf-8



import math
import os
import numpy as np




def square(x):
    return x * x




def classify_number(n):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"




assert classify_number(5) == "positive"
assert classify_number(-3) == "negative"
assert classify_number(0) == "zero"




results = []
for i in range(5):
    results.append(square(i))
print(results)




def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

