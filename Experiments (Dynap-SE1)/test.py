import sys
sys.path.append('/home/class_NI2021/ctxctl_contrib_2023')
import samna
import samna.dynapse1 as dyn1
from dynapse1constants import *
import dynapse1utils as ut
import netgen as n
import params
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

# Checking the list of unopened devices
devices = samna.device.get_unopened_devices()

if len(devices) == 0:
    raise Exception("no device detected!")

for i in range(len(devices)):
    print("["+str(i)+"]: ", devices[i], "serial_number", devices[i].serial_number)

# Select one device from the list
model,no_gui = ut.open_dynapse1(gui=False, sender_port=16254, receiver_port=16223, select_device=True) # returns Dynapse1Model

# remeber to close the device
samna.device.close_device(model)