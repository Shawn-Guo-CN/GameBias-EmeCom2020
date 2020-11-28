import re
import os
import sys

import numpy as np

#(vs, zdim) - (topo, posdis)

#(10, 5) -([0.38, 0.26,  0.29], [0.06, 0.04,  0.05])
#(10, 10) - ([0.23, 0.2, 0.2 ],[0,03, 0.05, 0.05 ])
#(10, 50) - ([0.09, 0.07, 0.08], [0.028, 0.03, 0.04])
# (50, 5) - ([], [])
# (50, 10) - ([], [])
# (50, 50) - ([], [])
# (60, 5) - ([], [])
# (60, 10) - ([], [])
# (60, 50) - ([], [])

def process_file(filename):
    with open(filename, 'r') as f:
        file_contents = ''.join(f.readlines())
    new_entry = {}
    z_dim = re.findall('z_dim=[0-9]*', file_contents)[0].split('=')[1]
    vocab_size = re.findall('vocab_size=[0-9]*', file_contents)[0].split('=')[1]
    beta = re.findall('beta=[0-9]*', file_contents)[0].split('=')[1]
    random_seed = re.findall('random_seed=[0-9]*', file_contents)[0].split('=')[1]
    new_entry['z_dim'] = int(z_dim)
    new_entry['vocab_size'] = int(vocab_size)
    new_entry['beta'] = int(beta)
    new_entry['random_seed'] = int(random_seed)
    topsim = []
    posdis = []
    for entry in re.findall('{"topsim": .*}', file_contents):
        topsim.append(eval(entry)['topsim'])
    for entry in re.findall('{"posdis": .*}', file_contents):
        posdis.append(eval(entry)['posdis'])
    new_entry['topsim'] = topsim
    new_entry['posdis'] = posdis
    return new_entry

def main(filepath):
    computed_result = []
    print(filepath)
    for filename in os.listdir(filepath):
        print('Processing file: {}'.format(filename))
        new_entry = process_file(os.path.join(filepath, filename))
        computed_result.append(new_entry)
    print(computed_result)

if __name__ == '__main__':
    main(sys.argv[1])