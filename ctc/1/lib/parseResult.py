import os,sys, math
import json
dirtyWordsSet = set()

def initDirtySet(path):
    if os.path.isfile(path):
        for line in open(path).readlines():
            if not line:
                continue
            item = line.strip()
            arr = item.split()
            dirtyWordsSet.add(' '.join(arr))

def parse(message):
    try:
        obj =json.loads(message)
        if 'result' in obj.keys():
            wordsArr = obj['result'].split()
            wordsLen = len(wordsArr)
            for i in range(wordsLen):
                if i < wordsLen-1:
                    if wordsArr[i] + ' ' + wordsArr[i+1] in dirtyWordsSet or  wordsArr[i] + wordsArr[i+1] in dirtyWordsSet:
                        wordsArr[i] = "".join(['*']*len(wordsArr[i]))
                        wordsArr[i+1] = "".join(['*']*len(wordsArr[i+1]))
                    elif wordsArr[i] in dirtyWordsSet:
                        wordsArr[i] = "".join(['*']*len(wordsArr[i]))
                else:
                    if wordsArr[i] in dirtyWordsSet:
                        wordsArr[i] = "".join(['*']*len(wordsArr[i]))
            obj['result'] = ' '.join(wordsArr)
            return json.dumps(obj)
    except Exception as e:
        return message
    return message
