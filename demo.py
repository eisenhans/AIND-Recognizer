# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:49:36 2018

@author: Markus
"""

import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame
print('asl head: {}'.format(asl.df.head()))

