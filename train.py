import torch 
import torch.nn as nn 

import datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import wordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['transalaton'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(wordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizers = Whitespace() 
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path)) 
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer 

def get_ds(cofig):
    ds_raw = load_dataset("opus_books", f'{config["src_lang"]}-{config["tgt_lang"]}', split = "train") 

    #build tokenizer 
    tokenzier_src = get_or_build_tokenizer(config, ds_raw, config["src_lang"]) 
    tokenzier_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt_lang"])

    #keep 90% for training  
    train_ds_size = int(len(ds_raw) * 0.9) 
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])    
