import os
from io import open

import torch

class Dictionary(object):

        def __init__(self):
            self.word2idx = [] # word -> index
            self.idx2word = [] # index -> word

        def add_word(self, word):
            if word not in self.word2idx: # word가 word2idx에 없을 경우
                self.idx2word.append(word) # idx2word에 그 단어 추가
                self.word2idx[word] = len(self.idx2word) - 1 # todo error뜨는데
                print(self.idx2word)
            return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)

class Corpus(object): # 말뭉치
    # token을 사람이 알아먹을 수있는 최소의 형태라고 생각하자
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # dictionary에 words를 추가
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                words = line.split() + ['<eos>'] # todo eos 의미?
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r', encoding='utf8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
















