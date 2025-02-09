🎭 Movie Review Sentiment Classifier 🎬
=======================================

*A Machine Learning-based Sentiment Analysis Tool*

📌 Overview
-----------

This project implements **Naive Bayes** and **Logistic Regression** models to classify movie reviews as **positive or negative**. The program processes a dataset of labeled movie reviews, extracts key features, and applies machine learning techniques to make predictions.

Originally tested on **IMDb movie reviews**, this system can be **generalized** to analyze sentiment in **any** textual dataset where positive/negative labels are applicable.

* * * * *

🛠 Features
-----------

✅ **Naive Bayes Classifier**: Uses probability-based classification with add-1 smoothing\
✅ **Logistic Regression Model**: Learns sentiment patterns for classification\
✅ **Custom Feature Engineering**: Supports unigram, bigram, and TF-IDF representations\
✅ **Efficient Text Processing**: Handles large datasets with optimized memory usage\
✅ **Customizable for Any Dataset**: Works with any dataset formatted as labeled text

* * * * *

🚀 How It Works
---------------

1️⃣ **Preprocessing**:

-   Reads and tokenizes movie reviews into words.
-   Removes stopwords and irrelevant symbols.

2️⃣ **Feature Extraction**:

-   Unigram features (single words).
-   Bigram features (word pairs).
-   TF-IDF weighting for better sentiment representation.

3️⃣ **Model Training**:

-   Naive Bayes: Uses word frequency probabilities to classify reviews.
-   Logistic Regression: Learns patterns in labeled training data.

4️⃣ **Prediction**:

-   Given a new review, the model predicts whether it's **positive or negative**.

5️⃣ **Evaluation**:

-   Reports accuracy on training, development, and test datasets.

* * * * *

📊 Example Dataset
------------------

-   **Input Format:**\
    The dataset consists of **movie reviews labeled as positive (1) or negative (0)**.\
    Example from `train.csv`:

    | Review | Label |
    | --- | --- |
    | "This movie was amazing!" | 1 |
    | "A complete disaster, boring and slow." | 0 |
    | "One of the best films I've seen in years." | 1 |

-   **Query Example (Using CLI)**

    `python main.py --model NaiveBayes`  
    `python main.py --model LogisticRegression`

-   **Output (Sample Predictions)**

    `Predicted Sentiment: Positive`  
    `Predicted Sentiment: Negative`

* * * * *

📦 Installation & Setup
-----------------------

### 1️⃣ Clone the Repository

`git clone https://github.com/your-repo/Movie-Sentiment-Classifier.git`  
`cd Movie-Sentiment-Classifier`

### 2️⃣ Install Dependencies

`pip install -r requirements.txt`

### 3️⃣ Run the Classifier

`python main.py --model NaiveBayes`  
`python main.py --model LogisticRegression`

For custom datasets, replace the input file with your own labeled dataset formatted like `train.csv`.

* * * * *

🧠 Models Implemented
---------------------

### 🔹 **Naive Bayes Classifier**

A probabilistic model that calculates the likelihood of a review being positive or negative based on word frequencies. Implemented with **Laplace Smoothing** to handle unseen words.

### 🔹 **Logistic Regression**

A linear model that learns **weights for words** to predict sentiment more effectively. Regularization options are included to **prevent overfitting**.

* * * * *

📂 File Structure
-----------------

- 📦 Movie-Sentiment-Classifier
- ├── 📄 classifiers.py      # Contains Naive Bayes & Logistic Regression implementations
- ├── 📄 utils.py           # Helper functions for text processing
- ├── 📄 main.py            # Main script to train and evaluate models
- ├── 📄 test.py            # Unit tests
- ├── 📂 data/
- │   ├── 📄 train.csv      # Training dataset
- │   ├── 📄 test.csv       # Testing dataset
- │   ├── 📄 dev.csv        # Development dataset
- ├── 📄 README.md          # This file`

* * * * *

📊 Performance & Evaluation
---------------------------

The models were tested on **IMDb movie reviews dataset**:

| Model | Train Accuracy | Dev Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Naive Bayes | 85% | 78% | 77% |
| Logistic Regression | 90% | 82% | 81% |

✅ **Observations**:

-   **Naive Bayes** works well for common words but struggles with context.
-   **Logistic Regression** performs better due to **weight learning** for word importance.
-   **Adding TF-IDF features** further improves accuracy.

* * * * *

🎯 Future Improvements
----------------------

🔹 Use **Deep Learning (LSTMs, Transformers)** for better performance\
🔹 Extend to **multi-class sentiment analysis** (e.g., emotions)\
🔹 Improve feature extraction with **word embeddings (Word2Vec, BERT)**

* * * * *

💡 Conclusion
-------------

✨ **A simple yet powerful text classification system** for **sentiment analysis**!\
🚀 Works on IMDb movie reviews but can **easily be adapted** to other datasets.\
🔍 Try it on **product reviews, tweets, or any text data**!
