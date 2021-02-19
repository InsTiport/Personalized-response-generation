import spacy

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


def tokenize(raw_text):
    """
    raw_text: raw text string
    tokens: a list of tokens of a tokenized sentence
    """

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
