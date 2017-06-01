"""Given that the datasets have been relocated, this script replaces the paths indicating the location of data with a specified new path.
For example: in testset.csv the script may replace /Documents/BAproject/Merged_filled_fixedsize/ with the new path ../data/sclera/
Usage: python relocate.py dataset.csv old_path new_path
"""

import sys
import csv

fname = sys.argv[1]
old_path = sys.argv[2]
new_path = sys.argv[3]

old_paths = [line for line in open(fname)]
for path in old_paths:
    path = path.lstrip(old_path)
    path = "".join([new_path, path])

new_paths = ["".join([new_path, path.lstrip(old_path)]) for path in old_paths]

with open(fname.split('.')[0] + "_new.csv", 'wb') as new_datafile:
    for path in new_paths:
        new_datafile.write(path)
