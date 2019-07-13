#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        pad_token_idx = self.target_vocab.char2id['<pad>']
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=pad_token_idx)

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_emb = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(input_emb, dec_hidden)
        scores = self.char_output_projection(output)
        
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        scores, _ = self.forward(char_sequence, dec_hidden) #(length, batch, self.vocab_size)
        #calculate p with softmax
        P = torch.nn.functional.log_softmax(scores, dim=-1)
        #filter paddings
        target_masks = (char_sequence != self.target_vocab.char2id['<pad>']).float()
        # get loss for every true label and sum them > probably with torch.gather
        target_gold_chars_log_prob = torch.gather(P, index=char_sequence.unsqueeze(-1), dim=-1).squeeze(-1) * target_masks
        
        loss = -target_gold_chars_log_prob.sum()
        
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        decodedWords = []
        
        start_char_id = self.target_vocab.char2id["{"]
        end_char_id = self.target_vocab.char2id["}"]
        
        batch_size = initialStates[0].size(1)
        output_words = torch.zeros((batch_size, max_length), dtype = torch.long)
        dec_hidden = initialStates
        
        #predict next character for each timestep
        for i in range(0, max_length):
            current_char_ids = torch.tensor([start_char_id]*batch_size, dtype=torch.long).view(1, -1) # start token for batches (1, batch)
            scores, dec_hidden = self.forward(current_char_ids, dec_hidden) #scores = (1, batch, self.vocab_size)
            probs = torch.nn.functional.softmax(scores, dim=-1) # (1, batch, self.vocab_size)
            current_char_ids = probs.argmax(dim=-1) #(1, batch)
            output_words[:, i] = current_char_ids.squeeze(0)
        
        # turn chars into words with getting rid of end token
        
        for word in output_words.tolist():
            filtered_word = ""
            for char_id in word:
                if char_id == end_char_id:
                    break
                filtered_word += self.target_vocab.id2char[char_id] 
            decodedWords.append(filtered_word) 
            
        return decodedWords        

        ### END YOUR CODE

