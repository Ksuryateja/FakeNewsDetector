
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchtext import data
from torchtext.data import BucketIterator
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.optim as optim

def Embedding(batch_size = 32, model='normal'):

    

    TEXT = data.Field(
        sequential=True,
        tokenize='spacy',
        lower=True,
        fix_length=100,
        batch_first = True
    )
    LABEL = data.Field(sequential = False, use_vocab = False, dtype=torch.float)
    
    if model == 'normal':
        path = './dataset/normal/'
        fields = [('labels', LABEL), ('statement', TEXT)]
    else:
        path = './dataset/attention/'
        fields = [('labels', LABEL), ('statement', TEXT), ('justification', TEXT)]
    
    train, val, test = data.TabularDataset.splits(
        path= path, format='csv', skip_header=True,
        train='train.csv', validation='val.csv', test = 'test.csv',
        fields=fields
    )

        
    TEXT.build_vocab(
        train, val, test,
        max_size=25000,
        unk_init = torch.Tensor.normal_,
        vectors=GloVe(name='6B', dim=50)
    )
    
    vocab_size = len(TEXT.vocab)
    word_embeddings = TEXT.vocab.vectors
    
    train_iter, val_iter, test_iter = data.Iterator.splits((train, val, test), sort_key=lambda x: len(x.statement), batch_sizes=(batch_size, batch_size, batch_size), device=torch.device('cuda'if torch.cuda.is_available() else 'cpu') )
    
    return train_iter, val_iter, test_iter, vocab_size, word_embeddings