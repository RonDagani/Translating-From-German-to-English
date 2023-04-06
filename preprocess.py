from torch.utils.data import Dataset, DataLoader
import torch
import re
import numpy as np
from gensim import downloader
import re
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch import nn
import torch
import os
import torch.nn.functional as F
from gensim.models import KeyedVectors

torch.manual_seed(25)


class LANG_Dataset(Dataset):
    def __init__(self, data_path, flag_train, word_indexers=None):
        """
        The function creates and initializes the paramterers of the dataset.

        Parameters
        ----------
        data_path: The path of the data
        word_indexers: Maps words to index in total words count, if not none use the input word_indexer

        """
        self.sen_english, self.sen_german, sen_english_sub, sen_german_sub = [], [], "", ""
        self.roots_english, self.modifiers_english = [], []
        self.data_path = data_path

        with open(self.data_path, 'rb') as f:
            lines = [x.decode('utf8').strip() for x in f.readlines()]
            for line in lines:
                if line == "German:":
                    language = 'german'
                    if len(sen_english_sub) != 0:

                        self.sen_english.append(sen_english_sub)
                    sen_english_sub = ""

                elif line == "English:":
                    language = 'english'
                    self.sen_german.append(sen_german_sub)
                    sen_german_sub = ""

                elif "Roots in English:" in line:
                    cur_roots = []
                    for root in line.split(": ")[1].split(","):
                        cur_roots.append(root)
                    self.roots_english.append(cur_roots)
                    self.sen_german.append(sen_german_sub)
                    sen_german_sub = ""

                elif "Modifiers in English:" in line:
                    all_modifiers = [value[0] for value in [x.split(")") for x in line.split(":")[1].split("(")][1:]]
                    self.modifiers_english.append(all_modifiers)

                elif line.strip():
                    if language == 'english':
                        sen_english_sub += ' '+ line
                    else:
                        sen_german_sub += ' '+ line
            self.sen_english.append(sen_english_sub)




