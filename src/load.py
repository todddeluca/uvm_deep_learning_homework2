'''
Functions for loading datasets from the data dir
'''

import pandas as pd
import config

def load_names():
    females = pd.read_csv(config.female_path, skiprows=6, header=None, names=['name'])
    females['sex'] = 'female'
    males = pd.read_csv(config.male_path, skiprows=6, header=None, names=['name'])
    males['sex'] = 'male'

    return pd.concat([females, males], ignore_index=True)

def load_jokes():
    '''return a dataframe with columns: body, category, id, title.  body contains the text of the joke.'''
    df = pd.read_json(config.jokes_path)
    return df

def load_pride_and_prejudice():
    '''return an unprocessed hunk of text with header lines prepended.'''
    with open(config.pnp_path) as fh:
        text = fh.read()
        return text
    