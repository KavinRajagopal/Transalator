from model import build_transformer
from dataset import BilingualDataset, casual_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    #precompute the encoder output and reuse it for every token we get from the decoder 
    encoder_output = model.encode(source, source_mask)
    #initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while true:
        if decoder_input.size(1) >= max_len:
            break

        #build mask for the target
        decoder_mask = casual_mask(decoder_input.size(1)).to(device)

        #calculate the output of the decoder 
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #get the next token
        prob = model.project(out[:,-1])
        #select the token with the maximum probability
        _, next_token = torch.max(prob, dim =1)
        #select the token with the maximum probability 
        decoder_input = torch.cat([decoder_input, torch.empty(1,1),type_as(source).fill_(next_word.item()).to(device)], dim = 1)
        if next_token.item() == eos_idx:
            break
        
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg,global_state, writer ,num_examples= 2):
    model.eval()
    count = 0

    source_text = []
    expected = []
    predicted = []

    #size of the control window 
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out.text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #print to teh console 
            print_msg('-'*console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Expected: {target_text}')
            print_msg(f'Predicted: {model_out_text}')

            if count == num_examples:
                break 

def get_all_sentences(ds, lang):
    print(ds)
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizers = Whitespace() 
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path)) 
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer 

def get_ds(cofig):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['src_lang']}-{config['tgt_lang']}", split='train')

    #build tokenizer 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["src_lang"]) 
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt_lang"])

    #keep 90% for training  
    train_ds_size = int(len(ds_raw) * 0.9) 
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])

    max_len_src = 0 
    max_len_tgt = 0


    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["src_lang"]]).ids 
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Maximum length of source language: {max_len_src}')
    print(f'Maximum length of target language: {max_len_tgt}')


    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True) 
    val_dataloader = DataLoader(val_ds, batch_size = config["batch_size"], shuffle = True)


    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    #define model 
    device = torch.device('cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_tgt.token_to_id("[PAD]"), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc = f'Processing Epoch {epoch}', position = 0, leave = True)
        for batch in batch_iterator:
            model.train()


            encoder_input = batch['encoder_input'].to(device) # [batch_size, seq_len]
            decoder_input = batch['decoder_input'].to(device) # [batch_size, seq_len]

            encoder_mask = batch['encoder_mask'].to(device)   # [batch_size, 1, 1, seq_len]
            decoder_mask = batch['decoder_mask'].to(device)   # [batch_size, 1, seq_len, seq_len]


            #run the tensors through the transformers
            encoder_output = model.encode(encoder_input, encoder_mask) # [batch_size, seq_len, d_model]
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # [batch_size, seq_len, d_model]
            proj_output = model.project(decoder_output) # [batch_size, seq_len, vocab_tgt_len]


            label = batch['label'].to(device) # [batch_size, seq_len]



            #(batch_size ,seq_len, vocab_tgt_len) -> (batch_size * seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            #Log the loss in tensorboard 

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            #backpropagation
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)


        #save the model at the end of every epoch 
        model_filename = get_weights_file_path(config, f'{epoch:02d}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)







