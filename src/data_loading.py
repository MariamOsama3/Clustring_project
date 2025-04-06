from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from config import NEWSGROUP_CATEGORIES, WIKI_DATASET_PATH

def load_20newsgroups_data():
    print("Loading 20 newsgroups dataset for categories:")
    print(NEWSGROUP_CATEGORIES)
    
    data = fetch_20newsgroups(subset='all', categories=NEWSGROUP_CATEGORIES,
                             shuffle=False, remove=('headers', 'footers', 'quotes'))
    
    # Convert to DataFrame
    df = pd.DataFrame({'text': data.data, 'category': data.target})
    return df

def load_wiki_data():
    return pd.read_csv(WIKI_DATASET_PATH)