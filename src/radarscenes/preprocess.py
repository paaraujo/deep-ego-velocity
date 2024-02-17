""" Define training, validation and testing sets.
"""

import os
import json
import random

random.seed(17)

# Path to dataset
root = os.getcwd()
dataset = os.path.join(root, 'data', 'radarscenes', 'data')

# Reading sequences and splitting into 'training' and 'validation' sets
with open(os.path.join(dataset,'sequences.json')) as json_file:
    parsed = json.load(json_file)
    sequences = parsed['sequences']

training = []
validation = []
for k in sequences.keys():
    info = sequences[k]
    if info['category'] == 'train':
        training.append(k)
    elif info['category'] == 'validation':
        validation.append(k)
print(len(training), len(validation))

# Selecting some sequences to build a 'test' set
testing = sorted(random.sample(validation, k=len(validation)//2), key=lambda x: int(x.split('_')[1]))
print(testing)

# Creating custom `sequences.json` file
with open(os.path.join(dataset, 'sequences.json')) as json_file:
    parsed = json.load(json_file)
    sequences = parsed['sequences']
    for k in sequences.keys():
        if k in testing:
            info = sequences[k]
            info.update({'category':'test'})
            sequences.update({k:info})
    parsed.update({'n_sequences':parsed['n_sequences'], 'sequences':sequences})

json_object = json.dumps(parsed, indent=2)

with open(os.path.join(dataset,'my_sequences.json'), 'w') as json_file:
    json_file.write(json_object)