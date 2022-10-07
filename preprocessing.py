import numpy as np
from torch.utils.data import DataLoader
import torch

on_gpu = True
cuda = torch.device('cuda')
LABEL_TO_ID = {'O': 0, 'I-PER': 1, 'I-LOC': 2, 'I-ORG': 3, 'B-LOC': 4, 'B-PER': 5, 'B-ORG': 6, 'I-MISC': 7, 'B-MISC': 8}


class CustomDataSet(torch.utils.data.Dataset):
  def __init__(self, x, y):
    super(CustomDataSet, self).__init__()
    # store the raw tensors
    self._x = x
    self._y = y

  def __len__(self):
    # a DataSet must know it size
    return self._x.shape[0]

  def __getitem__(self, index):
    x = self._x[index, :]
    y = self._y[index, :]
    return x, y


# Read and preprocess data from the CoNLL dataformat
# Code for preprocessing adopted from: https://medium.com/analytics-vidhya/ner-tensorflow-2-2-0-9f10dcf5a0a

def read_and_preprocess(file):

    data_raw = open("UKP Masters Test/data/"+file)
    data_pp = []
    sentence = []
    for row in data_raw:
        if len(row)==0 or row.startswith('-DOCSTART-') or row[0]=="\n":
                if (len(sentence) > 0):
                  data_pp.append(sentence)
                  sentence = []
                continue
        splits = row.split('\t')
        sentence.append([splits[0],splits[-1].rstrip("\n")])
        
    if len(sentence) > 0 and len(sentence) < 60:
        data_pp.append(sentence)
        sentence = []
    return [sentence for sentence in data_pp if len(sentence)<60]


def get_global_wordset(*annotated_wordlists):
    """ 
    Builds a set of all lowered words and a set of all labels found in the annotated_wordlists and returns both these sets.
    Arguments:
        annotated_wordlists: one annotated_wordlist per dataset.
    Returns:
        Set over all words and all labels found in the wordlists
    """
    wordSet = set()
    # words and labels
    for dataset in annotated_wordlists:
        for sentence in dataset:
            for word, label in sentence:
                wordSet.add(word.lower())
    return wordSet


def createMatrices(data, word2idx_vocab, label2Idx):
    """ Translates words and labels of a given dataset (param data) into ids using the vocab and LABEL_TO_ID """
    sentences = []
    labels = []
    for sentence in data:
        wordIndices = []
        labelIndices = []
        for word, label in sentence:
            if word in word2idx_vocab:
                wordIdx = word2idx_vocab[word]
            elif word.lower() in word2idx_vocab:
                wordIdx = word2idx_vocab[word.lower()]
            else:
                wordIdx = word2idx_vocab['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label]) 
        sentences.append(wordIndices)
        labels.append(labelIndices) 
    return sentences, labels


def get_word2idx_vocab(wordset):
    """ Creates a word vocab of form {word: idx} """
    vocab = {}
    if len(vocab) == 0:
        vocab["PADDING_TOKEN"] = len(vocab)
        vocab["UNKNOWN_TOKEN"] = len(vocab)
    for word in wordset:
        vocab[word] = len(vocab)
    return vocab


def get_max_seq_length(*datasets):
    lengths=[]
    for dataset in datasets:
        for sent in dataset:
            lengths.append(len(sent))
    return max(lengths)


def pad_sents_and_labels(sentences, labels, max_length, word2idx_vocab, on_gpu):
    # Initialize numpy array with padding values in desired dimensions
    # Reference: https://cs230.stanford.edu/blog/namedentity/
    batch_data = word2idx_vocab["PADDING_TOKEN"]*np.ones((len(sentences), max_length))
    batch_labels = 0*np.ones((len(labels), max_length))

    # Copy values into the arrays
    for j in range(len(sentences)):
        cur_len = len(sentences[j])
        batch_data[j][:cur_len] = sentences[j]
        batch_labels[j][:cur_len] = labels[j]
    
    if on_gpu:
      return torch.cuda.LongTensor(batch_data, device=torch.device('cuda')), torch.cuda.LongTensor(batch_labels, device=torch.device('cuda'))
    else:
      return torch.LongTensor(batch_data), torch.LongTensor(batch_labels)


def prepare_emb_index_and_matrix(word2idx_vocab):
    # Loading glove embeddings
    embeddings_index = {}
    f = open("UKP Masters Test/data/glove.6B.50d.txt", encoding="utf-8")
    for line in f:
        values = line.strip().split(' ')
        word = values[0] # the first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') #50d vectors representing the word
        embeddings_index[word] = coefs
    f.close()

    embeddings_matrix = np.zeros((len(word2idx_vocab), 50)) # Word embeddings for the tokens
    for word,i in word2idx_vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
        #else:
            # Initialize a random vector, if glove representation cannot be found for a respective, unique word in the vocab
            #embeddings_matrix[i] = np.random.normal(scale=0.6, size=(50, ))
    return embeddings_matrix

def run_pp_pipeline(on_gpu):
    train = read_and_preprocess("train.conll")
    dev = read_and_preprocess("dev.conll")
    test = read_and_preprocess("test.conll")

    wordset = get_global_wordset(train, dev, test)
    word2idx_vocab = get_word2idx_vocab(wordset)

    train_sentences, train_labels = createMatrices(train, word2idx_vocab, LABEL_TO_ID)
    dev_sentences, dev_labels = createMatrices(dev, word2idx_vocab, LABEL_TO_ID)
    test_sentences, test_labels = createMatrices(test, word2idx_vocab, LABEL_TO_ID)

    max_seq_length = get_max_seq_length(train_sentences, dev_sentences, test_sentences)

    padded_train_sentences, padded_train_labels = pad_sents_and_labels(train_sentences, train_labels, max_seq_length, word2idx_vocab, on_gpu)
    padded_dev_sentences, padded_dev_labels = pad_sents_and_labels(dev_sentences, dev_labels, max_seq_length, word2idx_vocab, on_gpu)
    padded_test_sentences, padded_test_labels = pad_sents_and_labels(test_sentences, test_labels, max_seq_length, word2idx_vocab, on_gpu)

    train_data = CustomDataSet(padded_train_sentences, padded_train_labels)
    dev_data = CustomDataSet(padded_dev_sentences, padded_dev_labels)
    test_data = CustomDataSet(padded_test_sentences, padded_test_labels)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    embeddings_matrix = prepare_emb_index_and_matrix(word2idx_vocab)

    return word2idx_vocab, LABEL_TO_ID, embeddings_matrix, max_seq_length, train_dataloader, dev_dataloader, padded_dev_labels, test_dataloader, padded_test_labels