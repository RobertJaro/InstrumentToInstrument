from datetime import timedelta, datetime

import numpy as np
from dateutil.parser import parse

with open('hinode_ca.sh') as f:
    lines = np.array(f.readlines())

lines = np.array([l for l in lines if 'FGFOCUS' not in l])
new_lines = lines[:2]

dates = [parse(l[-23:-7].replace('_', 'T')) for l in lines[2:]]
date_diff = np.diff(dates, append=datetime.now())

new_lines = np.concatenate([new_lines, lines[2:][date_diff > timedelta(minutes=30)]])

with open('filtered_hinode_ca.sh', 'w') as f:
    f.writelines(new_lines)
