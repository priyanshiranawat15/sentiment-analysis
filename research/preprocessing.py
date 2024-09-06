import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random
from utils import extract_features, process_tweet, build_freqs

import re                  
import numpy as np                # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


print('Number of positive tweets', len(all_positive_tweets))
print('Number of negative tweets', len(all_negative_tweets))


print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))


# Declare a figure with a custom size
fig = plt.figure(figsize=(5, 5))

# labels for the two classes
labels = 'Positives', 'Negative'

# Sizes for each slide
sizes = [len(all_positive_tweets), len(all_negative_tweets)] 

# Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  

# Display the chart
# plt.show()

print('\033[92m' + all_positive_tweets[random.randint(0,5000)])

# print negative in red
print('\033[91m' + all_negative_tweets[random.randint(0,5000)])

freqs = build_freqs(all_positive_tweets, labels)
X = np.zeros((len(all_positive_tweets), 3))

# tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
#                                reduce_len=True)

# # tokenize tweets
# pos_list = []
# for tweet in all_positive_tweets:
#     tweet = tokenizer.tokenize(tweet)
#     pos_list.append(tweet)

# stopwords_english = stopwords.words('english') 

# clean_tweets_pos = []

# for words in pos_list: # Go through every word in your tokens list
#     clean_words = []
#     for word in words:
#         if (word not in stopwords_english and  # remove stopwords
#             word not in string.punctuation):  # remove punctuation
#             clean_words.append(word)
#     clean_tweets_pos.append(clean_words)

# neg_list = []
# for tweet in all_positive_tweets:
#     tweet = tokenizer.tokenize(tweet)
#     pos_list.append(tweet)

# stopwords_english = stopwords.words('english') 

# clean_tweets_neg = []
# for words in pos_list: # Go through every word in your tokens list
#     clean_words = []
#     for word in words:
#         if (word not in stopwords_english and  # remove stopwords
#             word not in string.punctuation):  # remove punctuation
#             clean_words.append(word)
#     clean_tweets_neg.append(clean_words)

# def count_tweets(cleaned_list, label):
#     tweet_count = 0
#     for tweet in cleaned_list:
#         if label in tweet:
#             tweet_count += 1
#     return tweet_count

# counter = [] 

# X_train = []
# X_train.append(clean_tweets_pos)
# X_train.append(clean_tweets_neg)
# print(clean_tweets_pos[0],counter[0])