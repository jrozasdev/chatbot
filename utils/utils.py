from collections import Counter

"""
Function to count unique words in a list of sentences.
"""
def count_unique_words(sentences):
    words = []
    for sentence in sentences:
        words += sentence.split()
    return len(Counter(words).keys())