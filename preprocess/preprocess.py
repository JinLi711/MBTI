import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

def split_up_posts (df):
    """
    Split the posts column into multiple rows for each post
    
    :param df: DataFrame of data
    :type  df: pandas.core.frame.DataFrame
    :returns: DataFrame of data where the posts are split
    :rtype:   pandas.core.frame.DataFrame
    """
    
    types = df['type']
    MBTI_types = []
    comments = []
    for i, row in enumerate (df['posts']):
        posts = row.split('|||')
        for post in posts:
            comments.append(post)
            MBTI_types.append(types[i])
    return (pd.DataFrame({
        'type':  MBTI_types,
        'posts': comments
    }))


def train_test(df, test_size=.2):
    """
    Form train and test set.

    :param df: Dataframe from the csv
    :type  df: pandas.core.frame.DataFrame
    :param test_size: Proportion of test size and validation size
    :type  test_size: float
    :returns: (Train set, validate set, test set)
    :rtype:   (pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame,
               pandas.core.frame.DataFrame)
    """

    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=42
    )
    for train_index, test_index in split.split(df, df['type']):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    split = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=42
    )
    for train_index, valid_index in split.split(strat_train_set, strat_train_set['type']):
        strat_train_set = df.loc[train_index]
        strat_valid_set = df.loc[valid_index]
    
    print ('Size of Train Set is: {}'.format(strat_train_set.shape[0]) )
    print ('Size of Validation Set is: {}'.format(strat_valid_set.shape[0]) )
    print ('Size of Test Set is: {}'.format(strat_test_set.shape[0]) )

    return (strat_train_set, strat_valid_set, strat_test_set)


def var_row(row):
    """
    Compute the variance of a string.

    :param row: string of words where each comment is seperated by |||
    :type  row: str
    :return: Variance of the words
    :rtype:  float
    """

    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)


def augment_data (df):
    """
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """
    
    # divide by 50 because there are 50 comments
    df['Average Words Per Comment'] = df['posts'].apply(
        lambda x: len(x.split())/50
    )
    df['Variance of Word Counts'] = df['posts'].apply(
        lambda x: var_row(x)
    )
    return df


def clean_text(text):
    """
    Clean the text
    
    :param text: the comments of one person
    :type  text: str
    :return: string of cleaned words
    :rtype:  str
    """
    
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = re.sub(r'\|\|\|', r' ', text) # remove |||
    text = re.sub(r'http\S+', r' <URL> ', text) # replace urls with <URL>
    text = re.sub(r'[.]+', r'.', text) # remove ...
    text = re.sub(r"[0-9]+", r" <NUM> ", text) # replace all numbers with <NUM>
    text = re.sub(r' +', r' ', text) # remove all excessive white spaces
    return text
    
    
def clean_posts (df):
    """
    Clean each posts in the dataframe.
    
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """
    
    df['Cleaned Posts'] = df['posts'].apply(
        lambda x: clean_text (x)
    )
    return df


def split_personalities (df):
    """
    Split the label into 4 different labels, one for each duality.
    
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """
    
    types = ['IE', 'NS', 'TF', 'JP']
    for (i, trait) in enumerate (types):
        df[trait] = df['type'].apply(lambda x: x[i])
    return df


def scale_counts (df):
    """
    Apply standard scalers on the columns with numbers

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    cols = df[['Average Words Per Comment', 'Variance of Word Counts']]
    trans = pipeline.fit_transform(cols)
    df['Average Words Per Comment Scaled'] = trans[:,0]
    df['Variance of Word Counts Scaled'] = trans[:, 1]
    return df


def process_text(df):
    """
    Pipeline for processing the text.
    
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :return: Dataframe with more columns
    :rtype:  pandas.core.frame.DataFrame
    """
    
    df = augment_data (df)
    df = clean_posts (df)
    df = split_personalities (df)
    df = scale_counts(df)
    
    return df


data = pd.read_csv('../data/mbti_1.csv')
#data = split_up_posts (data) # doing this turns out to make accuracy very low
train, valid, test = train_test(data)
train = process_text(train)
valid = process_text(valid)
test = process_text(test)