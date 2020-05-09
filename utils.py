import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def preprocess_tweet(tweet):
    """
    This function preprocesses each tweet.
    
    Parameters:
    tweet(str) : Tweet of each user.
    
    Returns:
    tweet(str) : Preprocessed Tweet
    """
    tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

def extract_features(tweet_dataset):
    """
    Feature extraction from the tweets.
    
    Parameters:
    tweet(str): All tweets.
    
    Returns:
    features(numpy array): feature vectors for training with the ML model.
    """
    tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") 
    features=tfv.fit_transform(np.array(tweet_dataset.tweet))

    return features


