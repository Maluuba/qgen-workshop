#!/usr/bin/python
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class NewsQAData(object):

    def __init__(self, filepath, max_words=None):
        
        print('Processing text dataset')

        question_texts, answer_texts = [], []

        for i, row in pd.read_csv(filepath).iterrows():
            try:
                question = row['question']
                stok, etok = [int(i) for i in row['answer_token_ranges'].split(':')]
                answer = ''.join(row['story_text'].split()[stok:etok])
            except:
                # Skip bad token ranges
                continue
            question_texts.append(question)
            answer_texts.append(answer)

        # finally, vectorize the text samples into a 2D integer tensor
        # Answers
        self.answer_tokenizer = Tokenizer(num_words=max_words)
        self.answer_tokenizer.fit_on_texts(answer_texts)
        answer_sequences = self.answer_tokenizer.texts_to_sequences(answer_texts)
        self.answer_word_index = self.answer_tokenizer.word_index
        print('Found %s unique tokens in the answers' % len(self.answer_word_index))
        self.answer_data = pad_sequences(answer_sequences, maxlen=None)

        # Questions
        self.question_tokenizer = Tokenizer(num_words=max_words)
        self.question_tokenizer.fit_on_texts(question_texts)
        question_sequences = self.question_tokenizer.texts_to_sequences(question_texts)
        self.question_word_index = self.question_tokenizer.word_index
        print('Found %s unique tokens in the questions' % len(self.question_word_index))
        question_data_sparse = pad_sequences(question_sequences, maxlen=None)

        # Blow out to one hot encoding per word
        self.question_data = np.zeros((question_data_sparse.shape[0],
                                        question_data_sparse.shape[1],
                                        len(self.question_word_index) + 1),
                                        dtype=np.bool)
        for t in range(question_data_sparse.shape[0]):
            for s in range(question_data_sparse.shape[1]):
                v_i = question_data_sparse[t, s]
                self.question_data[t, s, v_i] = 1

        # Invert question word index for fast lookup
        self.question_index_to_word_map = {}
        for word, ix in self.question_word_index.items():
            self.question_index_to_word_map[ix] = word

        print('Shape of answer data tensor:', self.answer_data.shape)
        print('Shape of question data tensor:', self.question_data.shape)

    def get_answer_question_data(self):
        return self.answer_data, self.question_data

    def get_question_vocab_size(self):
        return len(self.question_word_index) + 1
    
    def get_answer_vocab_size(self):
        return len(self.answer_word_index) + 1

    def get_question_word_index(self):
        return self.question_word_index
    
    def get_answer_word_index(self):
        return self.answer_word_index

    def encode_answers(self, texts):
        return pad_sequences(self.answer_tokenizer.texts_to_sequences(texts),
                                maxlen=self.answer_data.shape[1])

    def decode_questions(self, encoded_questions, remove_oov=False, remove_padding=False):
        max_indices_per_time = np.argmax(encoded_questions, axis=2)
        decoded_questions = []
        for x in range(encoded_questions.shape[0]):
            decoded_tokens = []
            for t in range(encoded_questions.shape[1]):
                word_ix = max_indices_per_time[x, t]
                if word_ix == 0 and not remove_oov:
                    decoded_tokens.append("-UNK-")
                else:
                    decoded_tokens.append(self.question_index_to_word_map[word_ix])

            if remove_padding:
                start, end = 0, len(decoded_tokens)
                for token in decoded_tokens:
                    if token != "-UNK-":
                        break
                    start += 1
                for token in decoded_tokens[::-1]:
                    if token != "-UNK-":
                        break
                    end -= 1
                decoded_tokens = decoded_tokens[start:end]
            
            decoded_questions.append(" ".join(decoded_tokens) + "?")
        
        return decoded_questions