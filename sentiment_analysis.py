import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import bigrams
from collections import Counter

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english')) - {'not', 'no', 'this'}  # Keep 'this' for context
lemmatizer = WordNetLemmatizer()

# Enhanced feature extraction with improved negation handling
def extract_features(words):
    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]

    # Handle negation: Mark words after "not" or "no" for the next 3 words or until punctuation
    features = Counter()
    negate = False
    negation_window = 0
    for i, token in enumerate(tokens):
        if token in ['not', 'no']:
            negate = True
            negation_window = 3  # Affect the next 3 words
            continue
        if negate and negation_window > 0:
            features[f"NOT_{token}"] = 2  # Increase weight of negated words
            negation_window -= 1
            if negation_window == 0:
                negate = False
        else:
            if token not in stop_words:
                features[token] += 1
            negate = False

    # Add bigrams for sentiment-bearing phrases only
    bigram_features = ['_'.join(bigram) for bigram in bigrams(tokens)
                       if all(word.isalpha() for word in bigram) and
                       any(word not in stop_words for word in bigram)]
    for bigram in bigram_features:
        if 'not' in bigram or 'no' in bigram:
            features[bigram] = 2  # Increase weight of negated bigrams
        else:
            features[bigram] += 1

    return features

# Load movie reviews dataset
positive_reviews = [(list(movie_reviews.words(fileid)), 'Positive')
                    for fileid in movie_reviews.fileids('pos')]
negative_reviews = [(list(movie_reviews.words(fileid)), 'Negative')
                    for fileid in movie_reviews.fileids('neg')]

# Use all data for training and testing
split = 900  # 900 reviews per class for training
train_set = [(extract_features(words), category)
             for (words, category) in positive_reviews[:split] + negative_reviews[:split]]
test_set = [(extract_features(words), category)
            for (words, category) in positive_reviews[split:] + negative_reviews[split:]]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier
print("Accuracy:", accuracy(classifier, test_set))

# Function to classify new text
def classify_text(text, debug=False):
    words = word_tokenize(text.lower())
    features = extract_features(words)
    if debug:
        print(f"Features for '{text}': {dict(features)}")
        # Show probabilities for debugging
        probs = classifier.prob_classify(features)
        print(f"Probabilities for '{text}': Positive={probs.prob('Positive'):.3f}, Negative={probs.prob('Negative'):.3f}")
    return classifier.classify(features)

# Test cases
test_texts = [
    "I love this product!",
    "This is terrible",
    "Amazing experience, highly recommend!",
    "I hate this so much",
    "It's okay, nothing special",
    "This is not good",
    "I absolutely adore this!",
    "The worst experience ever",
    "I love to hate this movie",
    "Not a great experience"
]

for text in test_texts:
    print(f"Text: {text} → Sentiment: {classify_text(text, debug=True)}")

# Show the most informative features
classifier.show_most_informative_features(10)