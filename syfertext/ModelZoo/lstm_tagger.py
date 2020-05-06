import torch.nn as nn
import torch.nn.functional as F

class LSTM_Tagger(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers,
                            bidirectional = True,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)

        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        #embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))
        
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #we use our outputs to make a prediction of what the tag should be
        #predictions = [sent len, batch size, output dim]
        predictions = self.fc(self.dropout(outputs))

        return predictions