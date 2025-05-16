import random
from typing import AnyStr, Tuple
from collections import Counter
import numpy as np
import html
from settings import *

import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from typing import List

import random
import pandas as pd
from typing import Tuple

def separate_data() -> Tuple[list, list, list, list, list, list]:
    filename = 'dataset/dataset.csv'
    df = pd.read_csv(filename, encoding='utf8')

    if df.shape[1] != 2:
        raise ValueError(f"CSV does not have two columns, it has {df.shape[1]} columns.")

    # 获取中英文列
    zh_lines = df.iloc[:, 0].astype(str).apply(html.unescape).tolist()
    en_lines = df.iloc[:, 1].astype(str).apply(html.unescape).tolist()

    # 保证一一对应
    paired = list(zip(zh_lines, en_lines))
    random.seed(114514)
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(n_total * 0.98)
    n_valid = int(n_total * 0.01)

    train = paired[:n_train]
    valid = paired[n_train:n_train + n_valid]
    test = paired[n_train + n_valid:]

    train_zh = [x[0] + '\n' for x in train]
    train_en = [x[1] + '\n' for x in train]
    valid_zh = [x[0] + '\n' for x in valid]
    valid_en = [x[1] + '\n' for x in valid]
    test_zh = [x[0] + '\n' for x in test]
    test_en = [x[1] + '\n' for x in test]

    with open('dataset/train_zh.txt', 'w', encoding='utf8') as f:
        f.writelines(train_zh)
    with open('dataset/train_en.txt', 'w', encoding='utf8') as f:
        f.writelines(train_en)
    with open('dataset/valid_zh.txt', 'w', encoding='utf8') as f:
        f.writelines(valid_zh)
    with open('dataset/valid_en.txt', 'w', encoding='utf8') as f:
        f.writelines(valid_en)
    with open('dataset/test_zh.txt', 'w', encoding='utf8') as f:
        f.writelines(test_zh)
    with open('dataset/test_en.txt', 'w', encoding='utf8') as f:
        f.writelines(test_en)

    return train_zh, train_en, valid_zh, valid_en, test_zh, test_en

def count_words(filenames) -> Counter:
    word_counter = Counter()
    for filename in filenames:
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                words = line.strip().split()
                word_counter.update(words)
    return word_counter

def vocab():
    files = [
        'dataset/train_zh.txt',
        'dataset/valid_zh.txt',
        'dataset/test_zh.txt',
        'dataset/train_en.txt',
        'dataset/valid_en.txt',
        'dataset/test_en.txt'
    ]
    word_counts = count_words(files)
    lst = [item for item in word_counts.most_common() if item[1] >= 5]
    with open('dataset/index.txt', 'w', encoding='utf8') as f:
        idx = 20 
        for word, count in lst:
            f.write(f'{word}\t{idx}\n')
            idx += 1

def train_bpe_model(input_files, model_prefix, vocab_size=VOCAB_SIZE):
    # Train a BPE model using SentencePiece
    input_str = ','.join(input_files)
    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=PAD,
        unk_id=UNK,
        bos_id=-1,
        eos_id=EOS
    )

def load_bpe_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

# Replace space-based tokenization with BPE in to_idx

def to_idx():
    sp = load_bpe_model('dataset/bpe.model')
    files = [
        'dataset/train_zh.txt',
        'dataset/valid_zh.txt',
        'dataset/test_zh.txt',
        'dataset/train_en.txt',
        'dataset/valid_en.txt',
        'dataset/test_en.txt'
    ]
    for filename in files:
        with open(filename, 'r', encoding='utf8') as f:
            lines = f.readlines()
        with open(filename.replace('.txt', '_idx.txt'), 'w', encoding='utf8') as f:
            for line in lines:
                ids = sp.encode(line.strip(), out_type=int)
                f.write(' '.join(map(str, ids)) + '\n')

class TranslateData(Dataset):
    """Dataset for translation task."""
    def __init__(self, zh_file, en_file): 
        """Initialize the dataset. 
        # Example Usage: 
        dataset = TranslateData('dataset/train_zh_idx.txt', 'dataset/train_en_idx.txt')
        """
        self.zh_lines: List[np.ndarray] = []
        self.en_lines: List[np.ndarray] = []
        with open(zh_file, 'r', encoding='utf8') as f:
            for line in f:
                self.zh_lines.append(np.fromstring(line.strip(), dtype=np.int32, sep=' '))
        with open(en_file, 'r', encoding='utf8') as f:
            for line in f:
                self.en_lines.append(np.fromstring(line.strip(), dtype=np.int32, sep=' '))
        
        assert len(self.zh_lines) == len(self.en_lines), f"Length mismatch: {len(self.zh_lines)} vs {len(self.en_lines)}"

    def __len__(self):
        return len(self.zh_lines) * 2 # 中译英和英译中各一半
    
    def __getitem__(self, idx):
        i = idx // 2
        direction = idx % 2
        if direction == 0:
            # 中译英
            src_tokens = self.zh_lines[i]
            tgt_tokens = self.en_lines[i]
            src_lang = ZH
            tgt_lang = EN
        else:
            # 英译中
            src_tokens = self.zh_lines[i]
            tgt_tokens = self.en_lines[i]
            src_lang = EN
            tgt_lang = ZH

        # 拼接格式：<src_lang> 源句 <tgt_lang> 目标句 <eos>
        input_ids = [src_lang] + src_tokens + [tgt_lang] + tgt_tokens + [self.eos_id]
        tgt_start = 1 + len(src_tokens)  # <src_lang> + src_tokens

        return {
            "input_ids": input_ids,
            "tgt_start": tgt_start
        }

def collate_fn(batch, pad_id=0):
    input_ids_list = [item["input_ids"] for item in batch]
    tgt_starts = [item["tgt_start"] for item in batch]
    max_len = max(len(ids) for ids in input_ids_list)

    input_ids_padded = [ids + [pad_id]*(max_len - len(ids)) for ids in input_ids_list]
    labels_padded = [ids[1:] + [pad_id]*(max_len - len(ids)) for ids in input_ids_list]

    loss_mask = []
    for idx, ids in enumerate(input_ids_list):
        mask = [0]*tgt_starts[idx] + [1]*(len(ids)-tgt_starts[idx]) + [0]*(max_len - len(ids))
        loss_mask.append(mask)

    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    labels = torch.tensor(labels_padded, dtype=torch.long)
    loss_mask = torch.tensor(loss_mask, dtype=torch.float)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask
    }

if __name__ == '__main__' :
    # separate_data()
    # vocab()
    input_files = [
        'dataset/train_zh.txt',
        'dataset/valid_zh.txt',
        'dataset/test_zh.txt',
        'dataset/train_en.txt',
        'dataset/valid_en.txt',
        'dataset/test_en.txt'
    ]
    train_bpe_model(input_files, model_prefix='dataset/bpe')
    to_idx()