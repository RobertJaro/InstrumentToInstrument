import argparse
from datetime import timedelta, datetime

import numpy as np
from dateutil.parser import parse


parser = argparse.ArgumentParser(description='Prepare Hinode download script')
parser.add_argument('--sh_path', type=str, help='the path to the sh file including all available observations.')
parser.add_argument('--filtered_path', type=str, help='path to the result sh file.')

args = parser.parse_args()

file_name = args.sh_path
filtered_file_name = args.filtered_path

with open(file_name) as f:
    lines = np.array(f.readlines())

lines = np.array([l for l in lines if 'FGFOCUS' not in l])
new_lines = lines[:2]

dates = [parse(l[-23:-8].replace('_', 'T')) for l in lines[2:]]
date_diff = np.diff(dates, append=datetime.now())

new_lines = np.concatenate([new_lines, lines[2:][date_diff > timedelta(minutes=30)]])

with open(filtered_file_name, 'w') as f:
    f.writelines(new_lines)
