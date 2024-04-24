#!/usr/bin/python
# encoding: utf-8
import chardet

def getUTF8Txt(txt):
    enc = chardet.detect(txt)
    enType =  enc['encoding']
    try:
        txt = unicode(txt, enType, errors='replace')
        txt = txt.encode('utf-8')
    except :
            return txt
    return txt
