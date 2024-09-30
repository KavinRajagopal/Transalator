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
        
class MultiHeadAttentionBlock(nn.Module):

    # we do linear transformation to improve context understanding of different tokens in a sentence
    # otherwise all the wo-rds will have the same represdentation. 
    # we do multiple heads to chunk the data and transpose it for multiple proicessing such that each row referes to an individual head

    def _init_(self, d_model:int, h:int, dropout: float):
        super()._init_()

        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0 #d_model must be divisible by h

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.dropout):
        d_k = query.shape[-1]


       # (batch, h, seq_len, d_k) -> (batch x h, seq_len, d_k)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim = -1) #(batch x h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q) # Dimension (batch x seq_len x d_model) -> (batch, seq_length x d_model )
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) ->(batch, seq_len, h, d_k) -> (batch, h , seq_len, d_k  )
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch x seq_len x d_model) -> (batch, seq_length x d_model )
        key = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        value = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)


        #(batch, h, seq_len, d_k) -> (batch x seq_len, h, d_k) -> (batch x seq_len x d_model)
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.h * self.d_k)

        #(batch x seq_len x d_model) -> (batch x seq_length x d_model)
        return self.w_o(x)
    

    def ResidualConnection(self):

        def _init_(self, dropout:float):
            super()._init_()
            self.dropout = nn.dropout(dropout)
            self.norm = LayerNormalization()

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        

class EncoderBlock(nn.Module):

    def _init_(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block:FeedForward, dropout: float) -> None:
        super()._init_()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModukeList([ResidualConnection(dropout)for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0(x, lambda x:self.self_attention_block(x,x,x, src_mask))]
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x 
    

class Encoder(nn.module):

    def _init_(self, layers: nn.moduleList):
        super()._init_()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
        


            
    


    

















