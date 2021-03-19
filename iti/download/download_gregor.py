n_samples = 50
file_path = '/gss/r.jarolim/data/gregor/gregor_files.txt'

with open(file_path) as f:
    lines = f.readlines()

lines = sorted([l[2:-1] for l in lines if l.endswith('_sd.fts\n')])
lines = lines[::len(lines) // n_samples]

print('scp -P 2222 rjarolim@minos.aip.de:/store/gregor/hifi/level1.0/\{%s\} ./' % ','.join(lines))
