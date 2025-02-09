from nltk.tokenize import regexp_tokenize
from collections import Counter, defaultdict
import math
import numpy as np

# Here is a default pattern for tokenization, you can substitute it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature:
    """Bigram feature extractor."""
    
    def __init__(self):
        self.vocab = {}
        self.idf = []
        self.total_documents = 0
    
    def fit(self, text_set):
        """Learn bigram vocabulary and IDF values from the training data."""
        bigram_doc_counts = defaultdict(int)
        self.total_documents = len(text_set)
        
        # Calculate bigram counts across the documents
        for text in text_set:
            # Convert text to a list of words if it's not already
            if not isinstance(text, list):
                tokens = text.split()
            else:
                tokens = text

            bigrams = set()
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                bigrams.add(bigram)
            
            for bigram in bigrams:
                bigram_doc_counts[bigram] += 1
        
        # Create the vocab dictionary with bigrams as keys and indices as values
        index = 0
        for bigram in bigram_doc_counts:
            self.vocab[bigram] = index
            index += 1
        
        # Compute the IDF values for each bigram and store in a list
        for bigram, count in bigram_doc_counts.items():
            idf_value = math.log(self.total_documents / (1 + count))
            self.idf.append(idf_value)
    
    def transform(self, text):
        """Convert a single text into a TF-IDF feature vector."""
        if not isinstance(text, list):
            tokens = text.split()
        else:
            tokens = text

        bigram_freqs = Counter()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            bigram_freqs[bigram] += 1
        
        dense_vector = [0.0] * len(self.vocab)
        
        # Compute TF-IDF values for each bigram in the vocab
        for bigram, count in bigram_freqs.items():
            if bigram in self.vocab:
                idx = self.vocab[bigram]
                tf = count / len(bigram_freqs)
                dense_vector[idx] = tf * self.idf[idx]
        
        return dense_vector
    
    def transform_list(self, text_set):
        """Convert a list of texts into TF-IDF feature vectors."""
        feature_vectors = []
        for text in text_set:
            feature_vector = self.transform(text)
            feature_vectors.append(feature_vector)
        return feature_vectors


class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
    def fit(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform(self, text):
        # Add your code here!
        raise Exception("Must be implemented")
    def transform_list(self, text_set):
        # Add your code here!
        raise Exception("Must be implemented")


        
