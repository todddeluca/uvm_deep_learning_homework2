'''
Functions for loading datasets from the data dir
'''

import pandas as pd
from pathlib import Path


def female_path(data_dir):
    return data_dir / 'female_names.txt'


def male_path(data_dir):
    return data_dir / 'male_names.txt'


def jokes_path(data_dir):
    return data_dir / 'stupidstuff.json'


def pnp_path(data_dir):
    return data_dir / 'pride_and_prejudice.txt'


def default_data_dir(data_dir):
    return data_dir if data_dir else Path('~/data/2018/uvm_deep_learning_homework2/').expanduser()


def load_names(data_dir=None):
    data_dir = default_data_dir(data_dir)
    females = pd.read_csv(female_path(data_dir), skiprows=6, header=None, names=['name'])
    females['sex'] = 'female'
    males = pd.read_csv(male_path(data_dir), skiprows=6, header=None, names=['name'])
    males['sex'] = 'male'

    return pd.concat([females, males], ignore_index=True)


def load_jokes(data_dir=None):
    '''return a dataframe with columns: body, category, id, title.  body contains the text of the joke.'''
    data_dir = default_data_dir(data_dir)
    df = pd.read_json(jokes_path(data_dir))
    return df


def load_pride_and_prejudice(data_dir=None):
    '''return an unprocessed hunk of text with header lines prepended.'''
    data_dir = default_data_dir(data_dir)
    with open(pnp_path(data_dir)) as fh:
        text = fh.read()
        return text
    