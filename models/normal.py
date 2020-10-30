import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchtext import data
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.optim as optim

class LiarNet(nn.Module):

    def __init__(self, vocab_size, embed_weights, embed_length = 300, batch_size = 32,
                 
                 statement_hidden_units = 50,
                 statement_num_lstm = 2,
                 statement_bidirectional = True,

                 dropout = 0.4                 
                
                ):
        super(LiarNet, self).__init__()

        self.batch_size = batch_size
        self.embed_length = embed_length
        self.num_dir = 2 if statement_bidirectional else 1
        self.vocab_size = vocab_size
        self.dropout = dropout
        #<--------------------------------------------------Embedding-------------------------------------------------->
        self.word_embeddings = nn.Embedding(vocab_size, embed_length)
        self.word_embeddings.weights = nn.Parameter(embed_weights, requires_grad = False)

        #<--------------------------------------------------Statement-------------------------------------------------->
        self.statement_hidden_units = statement_hidden_units
        self.statement_num_lstm = statement_num_lstm
        
        self.statement_lstm = nn.LSTM(
            input_size = self.embed_length,
            hidden_size = self.statement_hidden_units,
            num_layers = self.statement_num_lstm,
            batch_first = True,
            bidirectional =  statement_bidirectional,
            dropout = 0.5
        )

        
        
        # Fully-Connected Layer
        self.fc1 = nn.Linear(self.num_dir * self.statement_hidden_units,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_final = nn.Linear(64, 1)
        




    
    def forward(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_inputs = batch.statement
        seq_inputs = Variable(seq_inputs).to(device)
        
        text_input = self.word_embeddings(seq_inputs)
        text_output, (h_t, h_0) = self.statement_lstm(text_input)

        features = torch.cat((h_t[-1,:,:], h_t[-2,:,:]), dim = 1)
        out = F.relu(self.fc_dropout(self.fc1(features)))
        out = F.relu(self.fc_dropout(self.fc2(out)))
        out = self.fc_final(out)
        return out