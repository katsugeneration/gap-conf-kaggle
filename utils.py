import stanfordnlp
nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma,pos')


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


def get_same_word_index(string, word):
    """Return same word index(slice) list.
    Args:
        string (str): target sentence
        word (str): target word
    Return:
        words (word): stanfordnlp word object list parsed input string.
        indexes (slice): python slice object list responsible same word index.
    """
    word = [w.lemma for w in nlp(word).sentences[0].words]
    doc = nlp(string)
    words = []
    for s in doc.sentences:
        words.extend(s.words)

    indexes = []
    for i, w in enumerate(words):
        if w.lemma == word[0]:
            s = slice(i, i+len(word))
            if [a.lemma for a in words[s]] == word:
                indexes.append(s)

    return words, indexes
