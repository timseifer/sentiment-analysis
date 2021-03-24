#import the necessary modules

import praw
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re, string, random
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
stop_words = stopwords.words('english')
from nltk import NaiveBayesClassifier
from nltk import classify
from praw.models import MoreComments

from nltk.stem.wordnet import WordNetLemmatizer

#attach tags corresponding to the words syntax
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# remove unecessary characters from the string
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# create an instance of a reddit object
reddit = praw.Reddit(
            user_agent="(USERNAME)",
            client_id="YOUR ID",
            client_secret="YOUR SECRET",
            username="YOUR USERNAM",
            password="YOUR PASSWORD",
        )

#go to the RealEstates Subreddit
subreddit = reddit.subreddit('RealEstate')

hot_python = subreddit.search("chicago housing", limit=500)

# extract content from the subreddit - right now the title is what is included
my_tokens = []
custom_tokens = []
for submission in hot_python:
    text = submission.title
    custom_tokens = word_tokenize(text)
    my_tokens.append(remove_noise(custom_tokens, stop_words))
    # get comment level data commented out for now
    # for top_level_comment in submission.comments:
    #         if isinstance(top_level_comment, MoreComments):
    #             continue
    #         my_tokens.append(remove_noise(top_level_comment.body, stop_words))

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(my_tokens)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

# generate trained model for identifying sentiment of text. This uses twitter
#  data which mus be considered.
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

positive = 0
negative = 0

# run the tokens extracted from reddit on the trained sentiment analyzer
for i in my_tokens:
    characterization = classifier.classify(dict([token, True] for token in i))
    print(i, characterization)
    if characterization == "Positive":
        positive += 1
    elif characterization == "Negative":
        negative += 1


print("total positive responses from 500 queries: " + str(positive))
print("total negative responses from 500 queries: " + str(negative))


