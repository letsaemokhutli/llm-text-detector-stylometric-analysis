import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from lexical_diversity import lex_div as ld

nltk.download('punkt')

# Define functions to compute the metrics
def type_token_ratio(text):
    tokens = word_tokenize(text)
    types = set(tokens)
    return len(types) / len(tokens)

def hapax_legomena(text):
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)
    hapaxes = [word for word in freq_dist if freq_dist[word] == 1]
    return len(hapaxes)

def hapax_dislegomena(text):
    tokens = word_tokenize(text)
    freq_dist = FreqDist(tokens)
    dislegomena = [word for word in freq_dist if freq_dist[word] == 2]
    return len(dislegomena)

def advanced_lexical_metrics(text):
    tokens = word_tokenize(text)
    mtld = ld.mtld(tokens)
    hdd = ld.hdd(tokens)
    return mtld, hdd

# Normalize the metrics
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Analyze lexical richness and compute a composite score
def analyze_lexical_richness(text):
    # Compute the individual metrics
    ttr = type_token_ratio(text)
    hapax_legomena_count = hapax_legomena(text)
    hapax_dislegomena_count = hapax_dislegomena(text)
    mtld, hdd = advanced_lexical_metrics(text)

    # Set arbitrary min and max values for normalization
    ttr_min, ttr_max = 0.2, 0.8
    hapax_legomena_min, hapax_legomena_max = 0, 100
    hapax_dislegomena_min, hapax_dislegomena_max = 0, 50
    mtld_min, mtld_max = 20, 200
    hdd_min, hdd_max = 0.5, 1.5

    # Normalize the metrics
    ttr_norm = normalize(ttr, ttr_min, ttr_max)
    hapax_legomena_norm = normalize(hapax_legomena_count, hapax_legomena_min, hapax_legomena_max)
    hapax_dislegomena_norm = normalize(hapax_dislegomena_count, hapax_dislegomena_min, hapax_dislegomena_max)
    mtld_norm = normalize(mtld, mtld_min, mtld_max)
    hdd_norm = normalize(hdd, hdd_min, hdd_max)

    # Compute the composite score (weights can be adjusted if needed)
    composite_score = (ttr_norm + hapax_legomena_norm + hapax_dislegomena_norm + mtld_norm + hdd_norm) / 5
    return composite_score