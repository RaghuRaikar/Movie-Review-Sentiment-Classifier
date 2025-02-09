import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse


def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))


def read_data(path):
    train_frame = pd.read_csv(path + 'train.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, test_frame = read_data(args.path)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']


    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']


    if args.model == "AlwaysPredictZero":
        model = AlwaysPredictZero()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    elif args.model == 'BonusClassifier':
        model = BonusClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")


    start_time = time.time()
    model.fit(X_train,Y_train)
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)
    
    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))
'''
   # Load and prepare data
    train_frame, test_frame = read_data('./data/')
    feat_extractor = UnigramFeature()

    # Process training data
    tokenized_text = [tokenize(text) for text in train_frame['text']]
    feat_extractor.fit(tokenized_text)
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label'].values

    # Initialize and train classifier
    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, Y_train)

    # Ensure that we extract vocabulary after fitting the feature extractor
    vocab = list(feat_extractor.unigram.keys())  # Make sure this is after fit
    print("Vocabulary Loaded:", len(vocab))  # Debug: Confirm vocabulary is loaded

    # Debug: Check log_likelihoods shape and some entries
    print("Log Likelihoods Check:", hasattr(classifier, 'log_likelihoods'))

    word_ratios = {}
    for index, word in enumerate(vocab):
        # Check if both positive and negative indices exist
        if index < len(classifier.log_likelihoods[1]) and index < len(classifier.log_likelihoods[0]):
            p_w_given_pos = np.exp(classifier.log_likelihoods[1][index])
            p_w_given_neg = np.exp(classifier.log_likelihoods[0][index])
            word_ratios[word] = p_w_given_pos / p_w_given_neg
        else:
            print(f"Index {index} for word '{word}' is out of range in log_likelihoods.")

    # Calculate and display the word ratios
    most_positive_words = sorted(word_ratios.items(), key=lambda item: item[1], reverse=True)[:10]
    most_negative_words = sorted(word_ratios.items(), key=lambda item: item[1])[:10]

    print("Most Positive Words:", most_positive_words)
    print("Most Negative Words:", most_negative_words)
'''


if __name__ == '__main__':
    main()