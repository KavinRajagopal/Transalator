import torch 
import torch.nn as nn 

from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def casual_mask(seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).type(torch.int)
        return mask == 0
    
    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_token = self.tokenizer_src.encode(src_text).ids 
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_paddding_tokens = self.seq_len - len(enc_input_token) - 2 # two for sos and eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_token) - 1 # one for sos

        if enc_num_paddding_tokens < 0 or dec_num_padding_tokens < 0:
            return ValueError("Sequence length is too long")
        
        #add SOS and EOS tokens to raw text 
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_token, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        
        )

        #add SOS token for decoder input 

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        #ADD EOS token and this is the expected output from the decoder 

        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len
            "encoder_mask": [encoder_input != self.pad_token].unsqueeze(0).unsqueeze(0).int(), #seq_len
            "decoder_mask": [decoder_input != self.pad_token ].unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)) #seq_len
            "label": label

        }
    
    
    



        

