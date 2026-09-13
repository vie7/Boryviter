import csv
import os
from collections import defaultdict
import random
import cv2
import torch
from torch.utils.data import Dataset

"""
Unified dataset for Boryviter


Dataset and Model differences

WetlandBirds =  (x1, y1, x2, y2)
Mine = (x_center, y_center, w, h)

Math, basically anchor is trying to find the center of the object and change it w and h

w = x2 - x1
h = y2 - y1
x_center = x1 + w/2
y_center = y1 + h/2

Multi-species drops to single class
Behavior_id andd subject_id are unnecessary rn

Video frame and pixel_coords
Box turns into percentages of the frame

"""

