from email import utils
import pandas as pd
from classifiers import NaiveBayesClassifier
from utils import UnigramFeature
from main import read_data, tokenize
import numpy as np

# Define a function to transform texts to a feature matrix
def transform_texts_to_features(texts, feature_extractor):
    tokenized_texts = [tokenize(text) for text in texts]
    return feature_extractor.transform_list(tokenized_texts)

# Load the training and test sets
train_frame, test_frame = read_data('./data/')

# Initialize the UnigramFeature extractor and fit it with the training texts
unigram_feature_extractor = UnigramFeature()
unigram_feature_extractor.fit([tokenize(text) for text in train_frame['text']])

# Transform the training texts to a unigram feature matrix
X_train = transform_texts_to_features(train_frame['text'], unigram_feature_extractor)
Y_train = train_frame['label'].values

# Load the development set (assuming you have a dev.csv in the './data/' directory)
dev_df = pd.read_csv('./data/dev.csv')

# Transform the development texts to a unigram feature matrix
X_dev = transform_texts_to_features(dev_df['text'], unigram_feature_extractor)
Y_dev = dev_df['label'].values

# Initialize the Naive Bayes Classifier and fit it to the training data
classifier = NaiveBayesClassifier()
classifier.fit(X_train, Y_train)

# Predict the labels on the development set
predicted_labels = classifier.predict(X_dev)

# The vocabulary is the keys of the 'unigram' attribute of the UnigramFeature extractor
vocab = list(unigram_feature_extractor.unigram.keys())

# Proceed with analyzing random examples from the dev set and computing word ratios...

word_ratios = {}
for index, word in enumerate(vocab):
    # Calculate the ratio of probabilities for each word
    prob_word_given_pos = np.exp(classifier.log_likelihoods[1][index])
    prob_word_given_neg = np.exp(classifier.log_likelihoods[0][index])
    word_ratios[word] = prob_word_given_pos / prob_word_given_neg

# Sort words by the ratio of positive to negative probabilities
sorted_words = sorted(word_ratios.items(), key=lambda item: item[1], reverse=True)

# The 10 most positive words
most_positive_words = sorted_words[:10]

# The 10 most negative words
most_negative_words = sorted(sorted_words, key=lambda item: item[1])[:10]

print("Most Positive Words:", most_positive_words)
print("Most Negative Words:", most_negative_words)

# Load the data
train_frame, test_frame = read_data('./data/')
dev_df = pd.read_csv('./data/dev.csv')  # Assuming this path to the development set

# Initialize and prepare the feature extractor
feature_extractor = UnigramFeature()
# Prepare tokenized text for feature extraction
tokenized_train_text = [tokenize(text) for text in train_frame['text']]
feature_extractor.fit(tokenized_train_text)  # Fit the feature extractor with training data

# Prepare the development set for prediction
tokenized_dev_text = [tokenize(text) for text in dev_df['text']]
X_dev = feature_extractor.transform_list(tokenized_dev_text)
Y_dev = dev_df['label'].values  # Assuming 'label' is the column name for labels in your dev set

# Initialize the classifier and fit it with training data
classifier = NaiveBayesClassifier()
X_train = feature_extractor.transform_list(tokenized_train_text)
Y_train = train_frame['label'].values
classifier.fit(X_train, Y_train)

# Predict labels for the development set
predicted_labels = classifier.predict(X_dev)

# Display some random examples from the development set along with predictions and actual labels
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(X_dev), 5, replace=False)
for i in random_indices:
    print(f"Document: {dev_df.iloc[i]['text']}")  # Assuming 'text' is the column name for text data in your dev set
    print(f"Predicted label: {predicted_labels[i]}, Actual label: {Y_dev[i]}")
