import errno
import os
import re


wordNum = {'零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'}
wordMoney = {'块', '元', '角', '分', '毛', '点', '时', '秒'}
wordDay = {'年', '月', '日', '天'}
wordMeasure = wordMoney.union(wordDay).union({'百', '千', '万', '亿', '字', '笔', '个'})
wordMeasureNum = wordNum.union(wordMeasure)
wordLast = wordNum.union(wordMoney).union({'第二个', '傻', '研', '主播', '运动员'})
wordCur = wordMeasureNum.union({'逼', '研', '小', '金'})
wordPunc = '…“。\'！,？ʕ”:：；?，;."、!'
wordNoPunc = {'总'}
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

    ## 单个词直接返回加标点
    if len(txt_seq) == 1:
        ## 纯数字不加标点
        if txt_seq[0].isdigit():
            txt_with_punc = txt_seq[0]
        else:
            txt_with_punc = txt_seq[0] + " " + class2punc[predict[-1]]
        return txt_with_punc

    for i, word in enumerate(txt_seq):
        punc = class2punc[predict[i]]
        ## 月份前标签修正
        if i == 1 and len(re.sub('\d',"", txt_seq[0])) / len(txt_seq[0]) < 0.6:
            punc = " "
        elif i > 0 \
            and ((word and word[0] in wordCur \
            and txt_seq[i-1] \
            and (txt_seq[i-1] in wordLast \
            or re.findall('[0-9]',txt_seq[i-1][-1]))) \
            or txt_seq[i-1][-1] in wordPunc):
            punc = " "
        ## NUM + NUM 连续数字间 或 量词 + 数字 不加标点
        if i > 0 and \
            ((re.findall('[0-9]', word[0]) and \
            re.findall('[0-9]', txt_seq[i-1][-1])) or \
            (word[0] in wordNum and \
            ( txt_seq[i-1][-1] in wordMeasureNum) )):
            punc = " "

        ## 阿拉伯数字前，若在字典中，不加标点
        if i > 0 and \
            re.findall('[0-9]',word[0]) and \
            txt_seq[i-1] in wordCur:
            punc = " "

        ## 前后单个字，且后字在字典中，不加标点
        if i > 0 and \
            word[0] in wordNoPunc and \
            len(txt_seq[i-1]) == 1 :
            punc = " "

        if len(txt_seq) - i > 3 and i > 0:
            ## NUM + A + NUM + B 不加标点
            if re.findall('[0-9]', txt_seq[i-1]) and \
                word[0] in wordMoney and \
                re.findall('[0-9]', txt_seq[i+1]) and \
                txt_seq[i+2] in wordMoney :
                punc = " "

            ## NUM + A NUM + B 不加标点
            if re.findall('[0-9]', txt_seq[i-1]) and \
                word[0] in wordMoney and \
                any(c_ in txt_seq[i+1] for c_ in wordMoney) and \
                any(c_ in txt_seq[i+1] for c_ in wordNum) and \
                txt_seq[i+2] in wordNum:
                punc = " "

            ## A + NUM + B 不加标点
            if re.findall('[0-9]', word[0]) and \
                txt_seq[i-1] in wordMoney and \
                txt_seq[i+1] in wordMoney :
                punc = " "

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
