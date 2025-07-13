import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from tqdm import tqdm

def get_all_sentences(ds, lang: str):
    for item in ds:
        yield item['translation'][lang]

#importing or building a tokensier from HuggingFace's tokenizers library
def get_or_build_tokeniser(config, ds, lang: str):
    tokeniser_path = Path(config['tokeniser_file'].format(lang))
    if not Path.exists(tokeniser_path):
        tokeniser = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokeniser.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[EOS]","[SOS]"], min_frequency=2)
        tokeniser.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokeniser.save(str(tokeniser_path))
    else:
        tokeniser = Tokenizer.from_file(str(tokeniser_path))
        
    return tokeniser

#loading the dataset and tokeniser
def get_ds(config, validation_split_ratio: float = 0.1):
    ds = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')    #has 2 columns (id and translation(a dictionary with keys as language codes and values as the text in that language)))
    
    tokeniser_src = get_or_build_tokeniser(config, ds, config['lang_src'])
    tokeniser_tgt = get_or_build_tokeniser(config, ds, config['lang_tgt'])
    
    train_ds_size = int(len(ds) * (1-validation_split_ratio))
    val_ds_size = len(ds) - train_ds_size
    
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_size, val_ds_size])   #splitting the dataset into train and validation sets
    
    train_ds = BilingualDataset(train_ds, tokeniser_src, tokeniser_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds, tokeniser_src, tokeniser_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    #find the maximum length of the source and target languages
    for item in ds:
        src_ids = tokeniser_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokeniser_tgt.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")
    
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)
    
    return train_dataloader, val_dataloader, tokeniser_src, tokeniser_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    config = get_config()
    Path(config['model']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokeniser_src, tokeniser_tgt = get_ds(config)
    model = get_model(config, len(tokeniser_src.get_vocab()), len(tokeniser_tgt.get_vocab()))
    model.to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] +1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokeniser_src.token_to_id("[PAD]"), label_smoothing=0.1)
    
    for epoch in range(initial_epoch,config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            encoder_input = encoder_input.to(torch.long)
            decoder_input = decoder_input.to(torch.long)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            
            proj_output = model.project(decoder_output)
            
            label = batch['label'].to(device)
            
            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            batch_iterator.set_postfix({'loss':f"{loss.item():6.4f}"})
            
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, model_filename)
        
if __name__ == "__main__":
    train_model()
        
            

