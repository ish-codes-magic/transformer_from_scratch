import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokeniser_src, tokeniser_tgt, lang_src, lang_tgt, seq_len) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokeniser_src = tokeniser_src
        self.tokeniser_tgt = tokeniser_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        
        self.start_token = torch.tensor(tokeniser_src.token_to_id("[SOS]"), dtype=torch.int64)
        self.end_token = torch.tensor(tokeniser_src.token_to_id("[EOS]"), dtype=torch.int64)
        self.pad_token = torch.tensor(tokeniser_src.token_to_id("[PAD]"), dtype=torch.int64)
        
    #magic function to get the length of the dataset from the class ()
    def __len__(self):
        return len(self.ds)
    
    #magic function to get the item at a particular index from the class (https://www.kdnuggets.com/2023/03/introduction-getitem-magic-method-python.html)
    def __getitem__(self, index):
        src_target_pair = self.ds[index]  #a dictionary with keys as id and translation(a dictionary with keys as language codes and values as the text in that language) for some index
        
        src_text = src_target_pair["translation"][self.lang_src]  #text for source language
        tgt_text = src_target_pair["translation"][self.lang_tgt]  #text for target language
        
        src_tokens = self.tokeniser_src.encode(src_text).ids  #get array of all the token ids in the source language
        tgt_tokens = self.tokeniser_tgt.encode(tgt_text).ids  #get array of all the token ids in the target language
        
        #padding required as the model always expects the same length of input(see readme for more details)
        enc_num_padding_tokens = self.seq_len - len(src_tokens) - 2  #input has all the tokens in the source language, start token and end token, so subtracting (2+length of source tokens) from the sequence length to get the number of padding tokens required
        dec_num_padding_tokens = self.seq_len - len(tgt_tokens) - 1  #input has all the tokens in the target language, start token, so subtracting (1+length of target tokens) from the sequence length to get the number of padding tokens required
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:  #sanity check
            raise Exception("Sequence length exceeded!")
        
        #convert the entire input into token tensors [sos + source tokens + eos + padding tokens] 
        encoder_input = torch.cat(
            [
                self.start_token.reshape(1),
                torch.tensor(src_tokens, dtype=torch.int64),
                self.end_token.reshape(1),
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )          
        
        #convert decoder input into [sos + target tokens + padding tokens]
        decoder_input = torch.cat(
            [
                self.start_token.reshape(1),
                torch.tensor(tgt_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #convert expected decoder output into [target tokens + eos + padding tokens]
        label = torch.cat(
            [
                torch.tensor(tgt_tokens, dtype=torch.int64),
                self.end_token.reshape(1),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == decoder_input.size(0) == label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  #we want to mask the padding tokens, so we create a mask with 1s for all the non-padding tokens and 0s for all the padding tokens and then unsqueeze it to make it compatible with the shape of the input (1,1,seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  #we want to mask the padding tokens, so we create a mask with 1s for all the non-padding tokens and 0s for all the padding tokens and then unsqueeze it to make it compatible with the shape of the input (1,1,seq_len) and then we apply the causal mask to mask the future tokens (1, seq_len, seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text,
            "label" : label
        }
        
#function to create a causal mask to mask the future tokens
def causal_mask(size):
    
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0