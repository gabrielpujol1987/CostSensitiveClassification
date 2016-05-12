# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:38:32 2016

@author: GabrielLocal
"""

import RunBenchmark
from RunBenchmark import Dataset_Enum as dse

results = RunBenchmark.RunBenchmark(dataset_enum=dse.creditscoring1, numFolds=3)

print(results)