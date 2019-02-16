import stanfordnlp
nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize')


def charpos_to_word_index(string, pos):
    """Convert character position to word index.
    Args:
        string (str): target sentence
        pos (int): word start position counted by character
    Return:
        words (word): stanfordnlp word object list parsed input string.
        word_index (int): responsible word index in all sentence.
    """
    doc = nlp(string)
    words = []
    for s in doc.sentences:
        words.extend(s.words)

    doc = nlp(string[pos:])
    after_words = []
    for s in doc.sentences:
        after_words.extend(s.words)

    return words, len(words) - len(after_words)
