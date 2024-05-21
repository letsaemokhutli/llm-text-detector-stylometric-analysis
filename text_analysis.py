
import re
import string
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import words
import enchant

nltk.download('words')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

english_vocab = set(words.words())
word_list = set(words.words())
lemmatizer = WordNetLemmatizer()
d = enchant.Dict("en_US")
sid = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

#define all text analysis functions

def get_sentiment_label(text):
    sentiment_scores = sid.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def remove_punctuation_and_lowercase(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def lemmatize(word):
    return lemmatizer.lemmatize(word)

def get_misspelled_words(text):
       # Create a TextBlob object
       blob = TextBlob(text)
       # Identify misspelled words
       misspelled_words = [word for word in blob.words if word != TextBlob(word).correct()]
       print(misspelled_words)
       # Return the list of misspelled words
       return len(misspelled_words)

def identify_sentence_type(sentence):
    if sentence.endswith('.'):
        return "Declarative"
    elif sentence.endswith('?'):
        return "Interrogative"
    elif sentence.endswith('!'):
        return "Exclamatory"
    else:
        return "Unknown"
    
def count_declarative_sentences(text_column):
    declarative_count = 0
    for sentence in text_column:
        if identify_sentence_type(sentence) == "Declarative":
            declarative_count += 1
    return declarative_count

def count_interrogative_sentences(text_column):
    declarative_count = 0
    for sentence in text_column:
        if identify_sentence_type(sentence) == "Interrogative":
            declarative_count += 1
    return declarative_count

def count_exlamatory_sentences(text_column):
    declarative_count = 0
    for sentence in text_column:
        if identify_sentence_type(sentence) == "Exclamatory":
            declarative_count += 1
    return declarative_count

def count_unknown_sentences(text_column):
    declarative_count = 0
    for sentence in text_column:
        if identify_sentence_type(sentence) == "Unknown":
            declarative_count += 1
    return declarative_count

def count_words(sentence):
    words = sentence.split()
    return len(words)

def count_sentences(text):
    pattern = r'[.!?]+'
    return len(re.split(pattern, text))

def count_verbs(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    verb_count = 0
    for word, tag in tagged_words:
        if tag.startswith('VB'):
            verb_count += 1
    return verb_count

def entity_count(text):
    doc = nlp(text)
    entity_count = len(doc.ents)
    return entity_count