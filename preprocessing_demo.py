import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample text
text = "I love this product! It's amazing and I'm loving it."

# Tokenization
tokens = word_tokenize(text.lower())
print("Tokens:", tokens)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
print("After stopwords removal:", filtered_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("After lemmatization:", lemmatized_tokens)