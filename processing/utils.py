import spacy
import string


def tokenize(raw_text):
    """
    raw_text: raw text string
    tokens: a list of tokens of a tokenized sentence
    """
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(raw_text)
    tokens = list()
    for token in doc:
        token = token.text.lower().strip()
        if token != '':
            tokens.append(token)

    return tokens


def generate_vocab(tokens_list):
    """
    tokens_list: a list of tokenized sentences which are lists themselves
    vocab: a set of words in the vocabulary
    """
    vocab = set()
    for sentence in tokens_list:
        for token in sentence:
            vocab.add(token)

    return vocab


def generate_word_index_map(vocab):
    """
    vocab: a set of words in the vocabulary
    word2idx: a dictionary map a word to its index
    """
    word2idx = dict()
    index = 0
    for word in vocab:
        word2idx[word] = index
        index += 1
    idx2word = {index: word for word, index in word2idx.items()}

    return word2idx, idx2word


def generate_indexed_sentences(tokens_list, word2idx):
    """
    tokens_list: a list of tokenized sentences which are lists themselves
    indexed_sentences: a list of indexed sentences
    """
    indexed_sentences = list()
    for tokens in tokens_list:
        indexed_sentence = list()
        for token in tokens:
            indexed_sentence.append(word2idx[token])
        indexed_sentences.append(indexed_sentence)

    return indexed_sentences


def _is_name(s):
    if s.isupper():
        return True

    lower_count = 0
    for i in range(len(s)):
        if s[i] in string.ascii_lowercase:
            lower_count += 1
    if lower_count == 1:
        return True
    elif lower_count == 2 and 'de' in s:  # for AB de VILLIERS
        return True
    else:
        return False


def check_start_of_turn(s):
    s = s.strip()

    # Q. or A.
    if s[:2] == 'Q.' or s[:2] == 'A.':
        return [s[0], s[2:]]

    # other cases have a semicolon present
    if ':' not in s:
        return False
    semicolon_idx = s.index(':')
    if not _is_name(s[:semicolon_idx]):
        return False
    return [s[:semicolon_idx], s[semicolon_idx + 1:]]


def partial_match(name_to_match, names):
    """

    :rtype: int
    """
    name_to_match = name_to_match.lower()
    names = [name.lower() for name in names]

    for index, name in enumerate(names):
        for part in name_to_match.split():
            if part in name:
                return index

    return -1
