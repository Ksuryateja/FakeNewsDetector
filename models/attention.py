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

                 justification_hidden_units = 50,
                 justification_num_lstm = 1,
                 justification_bidirectional = True,

                 dropout = 0.4                 
                
                ):
        super(LiarNet, self).__init__()

        self.batch_size = batch_size
        self.embed_length = embed_length
        self.vocab_size = vocab_size
        self.num_dir = statement_num_lstm
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
        #<--------------------------------------------------Justification-------------------------------------------------->
        self.justification_hidden_units = justification_hidden_units
        self.justification_num_lstm = justification_num_lstm
        self.justification_num_direction = 2 if justification_bidirectional else 1
        
        self.justification_lstm = nn.LSTM(
            input_size = self.embed_length,
            hidden_size = self.justification_hidden_units,
            num_layers = self.justification_num_lstm,
            batch_first = True,
            bidirectional = justification_bidirectional,
            dropout = 0.5
        )

        
        
        # Fully-Connected Layer
        self.fc1 = nn.Linear(self.num_dir * self.statement_hidden_units,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_final = nn.Linear(64, 1)
        
    def apply_attention(self, lstm_output, final_state):
        """Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		"""
        hidden = final_state
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
        return new_hidden_state
    
    def forward(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #<--------------------------------------------------Statement-------------------------------------------------->
        statement = batch.statement
        statement = Variable(statement).to(device)
        
        statement_input = self.word_embeddings(statement)
        statement_output, (h_t, h_0) = self.statement_lstm(statement_input)
        #statement_output_final = torch.cat((statement_output[:,:,-1], statement_output[:,:,-2]), dim = 2)
        #<--------------------------------------------------Justification-------------------------------------------------->
        justification = Variable(batch.justification)
        justification = justification.to(device)

        justification_input = self.word_embeddings(justification)
        justification_output, (justification_hidden, justification_) = self.justification_lstm(justification_input)
        
        justification_final_hidden = torch.cat((justification_hidden[-1,:,:], justification_hidden[-2,:,:]), dim = 1)


        attention_out = self.apply_attention(statement_output, justification_final_hidden)
        out = F.relu(self.fc_dropout(self.fc1(attention_out)))
        out = F.relu(self.fc_dropout(self.fc2(out)))
        out = self.fc_final(out)
        return out