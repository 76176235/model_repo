import os
import sys

sys.path.append('/')
sys.path.append('../../../ctc/1/biaodian/nmt_punctuator_bin/')
from add_punctuation import usePunctuator

wordPara = 'word'

# engine = Sentence_Correction()
enSet = set()


def load(path, lan):
    global engine
    engine = usePunctuator(language=lan)
    try:
        engine.load_data(path)

        text = "let 's go to farmin"
        arr = text.split()
        result = engine.handle(arr)
        print(result)

    except Exception as e:
        print(e)


def get_input(wordArr, index):
    if index == 0:
        arr = []
        for i in range(0, min(2, len(wordArr))):
            arr.append(wordArr[i][wordPara])
        return arr, 0
    elif index == len(wordArr) - 1:
        arr = []
        for i in range(max(len(wordArr) - 1 - 2, 0), len(wordArr)):
            arr.append(wordArr[i][wordPara])
        return arr, len(arr) - 1
    else:
        arr = []
        for i in range(index - 1, index + 2):
            arr.append(wordArr[i][wordPara])
        return arr, 1


def handle(wordArr):
    global engine
    global enSet
    try:

        res_arr = engine.handle(wordArr)
        '''
        if wordStr != formatWord:
            print(wordStr, formatWord)
        '''
        return res_arr
    except Exception as e:
        print(e)
        return None


def handle_file(r_path="data", file_name="input.tok.txt", s_path="data", log_path="data"):
    engine.handle_file(r_path, file_name, s_path, log_path)


def test():
    global engine
    arr = []
    text = "it 's a gooddd night"
    arr.append(text)
    text = "it 's a transl night"
    text = "let 's go to farmin"
    arr.append(text)
    # arr = text.split()
    for i in range(len(arr)):
        print(arr[i])
        result = engine.handle(arr[i].split())
        print(result)


# if __name__ == "__main__":
#     load("/mnt/data/nmt_punctuation/punctuation_public", "en")
#     test()