import torch 
import torch.nn as nn


class InputEmbedding(nn.module):

    def _init_(self, d_model:int, vocab_size:int ):
        super()._init_()
        self.d_model = d_model
        self.vocab_Size = vocab_size
        self.embedding == nn.Embedding(vocab_size, d_model)


    def forward(self, x):
    #n the embedding layers, we multiply those weights by √d model.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def _init_(self, d_model:int, seq_len:int, dropout:float) -> None:
        super()._init_()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout( dropout)

        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a  vector of shape (seq,1)
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))


        #Apply the sin to even positions 
        pe[:,0::2] = torch.sin(position * div_term) 
        #Apply the cos to odd positions
        pe[:,1::2] = torch.cos(position *div_term)

        #add a batch dimension to psoition encoding 
        pe = pe.unsqueeze(0) #(1, Seq_len, d_model) 

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+ (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
    class LayerNormnalization(nn.module):
        
        def _init_(self, eps:float = 10**-6 ) -> None:
            super()._init_()
            self.eps = eps
            self.alpha == nn.Parameter(torch.ones(1)) #multiplied
            self.bias = nn.Parameter(torch.zeros(1)) #added

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std =x.std(dim = -1, keepdim = True)
            return self.alpha  * (x-mean) / (std + self.eps) + self.bias
        
class FeedForward(nn.Module):
        
        def  _init_(self, d_model:int,d_ff:int, dropout:float):
            super()._init_()
            self.linear_1 = nn.Linear(d_model, d_ff) #w1 and B1
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.linear(d_ff, d_model) #w2 aND b2

        def forward(self, x):
            #(batch x Seq_len, d_model) -> (batch x seq_len, d_ff) -> (batch x seq_len x d_model)
            return self.linear_2(self.dropout(F.relu(self.linear_1(x))))
        
        

            
    


    

















