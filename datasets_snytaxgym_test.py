#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:14:09 2025

@author: paveenhuang
"""

from datasets import load_dataset

ds = load_dataset("cpllab/syntaxgym", "all-2020")
ex = ds["test"][0]

print(ex)
print("FIELDS:", ex.keys())