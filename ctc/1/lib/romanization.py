import sys, math
from pythainlp.romanization import romanization as th_roman

head_jamos = [
    'g', #ㄱ
    'gg', #ㄲ
    'n', #ㄴ
    'd', #ㄷ
    'dd', #ㄸ
    'r', #ㄹ
    'm', #ㅁ
    'b', #ㅂ
    'bb', #ㅃ
    's', #ㅅ
    'ss', #ㅆ
    '',
    'j', #ㅈ
    'jj', #ㅉ
    'ch', #ㅊ
    'k', #ㅋ
    't', #ㅌ
    'p', #ㅍ
    'h' #ㅎ
]

body_jamos = [
    'a', #ㅏ
    'ae', #ㅐ
    'ya', #ㅑ
    'yae', #ㅒ
    'eo', #ㅓ
    'e', #ㅔ
    'yeo', #ㅕ
    'ye', #ㅖ
    'o', # ㅗ
    'wa', # ㅘ
    'wae', # ㅙ
    'oe', # ㅚ
    'yo', # ㅛ
    'u', # ㅜ
    'weo', # ㅝ
    'we', # ㅞ
    'wi', # ㅟ
    'yu', # ㅠ
    'eu', # ㅡ
    'eui', # ㅢ
    'i' # ㅣ
]

tail_jamos = [
    '',
    'g', # ㄱ
    'gg', # ㄲ
    'gs', # ㄱㅅ
    'n', # ㄴ
    'nj', # ㄴㅈ 
    'nh', # ㄴㅎ
    'd', # ㄷ
    'l', # ㄹ
    'rk', # ㄹㄱ
    'rm', # ㄹㅁ
    'rb', # ㄹㅂ
    'rs', # ㄹㅅ
    'rt', # ㄹㅌ
    'rp', # ㄹㅍ
    'rh', # ㄹㅎ
    'm', # ㅁ
    'b', # ㅂ
    'bs', # ㅂㅅ
    's', # ㅅ
    'ss', # ㅆ
    'ng', # ㅇ
    'j', # ㅈ
    'ch', # ㅊ
    'k', # ㅋ
    't', # ㅌ
    'p', # ㅍ
    'h' # ㅎ
]
th_consonantArr=['ก-k',
        'ข,ฃ,ค,ฅ,ฆ-kh',
        'ง-ng',
        'บ-b',
        'จ,ฉ,ช,ฌ-ch',
        'ฎ,ด,ฑ-d',
        'ต,ฏ-t',
        'ฝ,ฟ-f',
        'ห,ฮ-h',
        'ล,ฬ-l',
        'ญ,ณ,น,ร,ล,ฬ-n',
        'ม-m',
        'บ,ป,ฟ-p',
        'ฐ,ฑ,ฒ,ถ,ท,ธ-t',
        'ผ,พ,ภ-ph',
        'ร-r',
        'ฏ,ต-t',
        'ซ,ศ,ษ,ส-s',
        'ว-w',
        'ย,ญ-y',
        'ว,อ-o'
        ]

th_consonantDict={}
for item in th_consonantArr:
    tmpArr= item.strip().split('-')
    if len(tmpArr) == 2:
        firstArr = tmpArr[0].split(',')
        for con in firstArr:
            th_consonantDict[con] = tmpArr[1]
def parse_special_th(word):
    arr = []
    for item in word:
        if item in th_consonantDict.keys():
            arr.append(th_consonantDict[item])
    return ''.join(arr)

def parse(text):
    if sys.version_info[0] == 2:
        text = unicode(text, 'utf-8')
    retval = u''
    ga = 0xac00
    hih = 0xd7a3
    interval_head = 588
    interval_body = 28
    last_char_is_hangul = False

    for c in text:
        cint = ord(c)
        if ga <= cint <= hih:
            head = int(math.floor((cint - ga) / interval_head))
            headl = int(math.floor((cint - ga) % interval_head))
            body = int(math.floor(headl / interval_body))
            tail = int(math.floor(headl % interval_body))
            if last_char_is_hangul:
                retval += ''
            retval += head_jamos[head]
            retval += body_jamos[body]
            retval += tail_jamos[tail]
            last_char_is_hangul = True
        else:
            last_char_is_hangul = False
            retval += c
    return retval

def handle_ko(sen):
    arr = sen.split()
    resultArr = []
    for word in arr:
        #if len(word[0]) == 1 and len(word[-1]) == 1:
        if word.encode('UTF-8').isalpha():
            resultArr.append(word)
        else:
            resultArr.append(parse(word))
    return ' '.join(resultArr)

def handle_th(sen):
    arr = sen.split()
    resultArr = []
    for word in arr:
        if word.encode('UTF-8').isalpha():
            resultArr.append(word)
        else:
            th_word= th_roman(word)
            if len(th_word.strip()) == 0:
                th_word = parse_special_th(word)
            resultArr.append(th_word)
    return ' '.join(resultArr)

