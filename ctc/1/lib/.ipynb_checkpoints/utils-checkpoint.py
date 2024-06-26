# Created on 2019/02
# Author: Kaituo XU (NPU-ASLP)

import errno
import os

IGNORE_ID=-1


def load_vocab(vocab_path, extra_word_list=[], encoding='utf8'):
    n = len(extra_word_list)
    with open(vocab_path, encoding=encoding) as vocab_file:
        vocab = { word.strip(): i + n for i, word in enumerate(vocab_file) }
    for i, word in enumerate(extra_word_list):
            vocab[word] = i
    return vocab

def num_param(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

def add_punc_to_txt(txt_seq, predict, class2punc):
    """Add punctuation to text.
    Args:
        txt_seq: text without punctuation
        predict: list of punctuation class id
        class2punc: map punctuation class id to punctuation
    Returns:
        txt_with_punc: text with punctuation, without newline
    """
    # txt_with_punc,num_list_idx,num_list_len = "",0,len(num_list)
    txt_with_punc = ""
    if isinstance(txt_seq,str):
        txt_seq = txt_seq.split()
    for i, word in enumerate(txt_seq):
        punc = class2punc[predict[i]]
#         if num_list_idx < num_list_len and word == "<NUM>":
#             word = num_list[num_list_idx]
#             num_list_idx +=1
        txt_with_punc += word + " " if punc == " " else punc + " " + word + " "
    punc = class2punc[predict[i + 1]]
    if punc != " ":
        txt_with_punc += punc
    return txt_with_punc


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    vocab = load_vocab(sys.argv[1], ["<UNK>", "<END>"])
    print(vocab)
    vocab = load_vocab(sys.argv[1])
    print(vocab)
