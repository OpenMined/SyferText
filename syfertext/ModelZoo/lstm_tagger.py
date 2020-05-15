import torch.nn as nn
import torch.nn.functional as F
import syft as sy

from .model import Model

hook = sy.TorchHook(torch)

class LSTM_Tagger(Model):
    def __init__(self, 
                 embeddings = None, 
                 input_dim, 
                 embedding_dim,
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 dropout, 
                 padding_idx):
        
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim =  embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        if embeddings not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.data[padding_idx] = torch.zeros(embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers,
                            bidirectional = True,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)

        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        #pass text through embedding layer
        #embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))
        
        
        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        #we use our outputs to make a prediction of what the tag should be
        #predictions = [sent len, batch size, output dim]
        predictions = self.fc(self.dropout(outputs))

        return predictions

    def loss(self, predictions, labels):

        #predictions = [sent len, batch size, output dim]
        #labels = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)

        criterion = nn.CrossEntropyLoss(ignore_index = self.padding_idx)

        #predictions = [sent len * batch size, output dim]
        #labels = [sent len * batch size]
        loss = criterion(predictions, labels)

    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""

        model_state = {
            "state_dict": self.state_dict(),
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "padding_idx" self.padding_idx
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary."""
        
        model = LSTM_Tagger(
            input_dim= state['input_dim'],
            embedding_dim = state['embedding_dim'],
            hidden_dim = state['hidden_dim'],
            output_dim = saet['output_dim'],
            n_layers = state['n_layers'],
            dropout = state['dropout'],
            padding_idx = state['padding_idx']
            )

        model.load_state_dict(state["state_dict"])

        return model


        


        
