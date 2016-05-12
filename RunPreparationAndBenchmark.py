# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:38:32 2016

This code is to facilitate the running of the benchmark.

@author: gabrielpujol87
"""

import RunBenchmark
from RunBenchmark import Dataset_Enum as dse

results = RunBenchmark.RunBenchmark(dataset_enum=dse.creditscoring1, numFolds=3, trainRatio=0.8)

print(results)