import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    def __init__(self):
        # Initialize dictionaries to store log probabilities
        self.log_prior = {}  # Store log prior probabilities of each label
        self.log_likelihoods = {}  # Store log likelihood probabilities for features given labels

    def fit(self, X, Y):
        # Determine the number of features in the dataset
        num_features = len(X[0])
        # Count the total number of documents
        num_docs = len(Y)
        # Dictionary to count occurrences of each label
        label_counts = {}

        # Loop through each label in the dataset
        for label in Y:
            # If the label is not already in the dictionary, initialize it
            if label not in label_counts:
                label_counts[label] = 0
            # Increment the count for this label
            label_counts[label] += 1

        # Calculate log prior probabilities for each label
        for label in label_counts:
            # Log of the probability of the label
            self.log_prior[label] = np.log(label_counts[label] / num_docs)

        # Calculate log likelihoods for each label
        for label in label_counts:
            # Initialize an array to sum feature counts for documents with the same label
            total_words = np.zeros(num_features)
            # Iterate over all documents and their corresponding labels
            for i, doc in enumerate(X):
                # If the document's label matches the current label, add the feature counts
                if Y[i] == label:
                    total_words += doc  # Sum feature counts for this label

            # Calculate the denominator for log likelihood computation, including smoothing
            denominator = sum(total_words) + num_features  # Add-1 smoothing

            # Calculate log likelihood for each feature under the current label
            self.log_likelihoods[label] = np.log((total_words + 1) / denominator)

    def predict(self, X):
        # List to hold predictions for each document
        predictions = []
        # Iterate over each document for classification
        for doc in X:
            # Dictionary to store scores for each label
            scores = {}
            # Calculate score for each label
            for label in self.log_prior:
                # Start with the log prior probability
                score = self.log_prior[label]
                # Add contribution from each feature in the document
                for j in range(len(doc)):
                    # Consider only non-zero features to save computation
                    if doc[j] > 0:
                        score += doc[j] * self.log_likelihoods[label][j]
                # Store the calculated score for this label
                scores[label] = score

            # Find the label with the highest score for this document
            best_label = None
            max_score = float('-inf')  # Initialize with the lowest possible float
            # Iterate through scores to find the maximum
            for label, score in scores.items():
                if score > max_score:
                    max_score = score
                    best_label = label
            # Append the label with the highest score to the predictions list
            predictions.append(best_label)

        # Convert predictions list to a numpy array and return
        return np.array(predictions)




# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
