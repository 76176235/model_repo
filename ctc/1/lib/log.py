#!/usr/bin/python
# encoding: utf-8
import logging
import logging.handlers
import time,datetime
import os
import traceback
from queue import Queue
from threading import Thread
import hashlib,urllib
import requests
import socket
import inspect
import sys
import json
sys.path.append('')
print(sys.path)

from lib import fplog

_ipAddress = ''
def get_host_ip():
    global _ipAddress
    if _ipAddress:
        return _ipAddress
    ip = ''
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception as e:
        return ''
    _ipAddress = ip
    print(_ipAddress)
    return ip

class LogInfoClass:
    def __init__(self):
        global lang_pair_name
        self.ltype = 'info'
        self.message = ''
        self.ip = get_host_ip()
        self.langPair = lang_pair_name
    def getJsonStr(self):
        return json.dumps(self.__dict__)


_fpLog = fplog.FPLog('/tmp/fplog.sock', conn_pool=3)
logError = "alt.punctuation.monitorLog.error"
logDebug = "alt.punctuation.monitorLog.debug"
logKibana = "alt.punctuation.remoteLog"

_isNeedSendLog = True
if  get_host_ip() in  '10.0.64.67 10.0.64.74'.split():
    _isNeedSendLog = False

def set_logServer_path(path):
    global _fpLog
    if path:
        _fpLog.__init__(path,conn_pool=3)

def send_to_log_server_file(senObj):
    global _fpLog
    if not _isNeedSendLog:
        return
    try:
        _fpLog.write('nmt.' + senObj.ltype, senObj.getJsonStr())
    except Exception as e:
        print(e)
        traceback.print_exc()

filename = 'test.log'
td = datetime.datetime.utcnow().strftime("%Y%m")
#filenameError = os.path.join(config.path_log, td + '.error.log')
#filenameInfo = os.path.join(config.path_log, td +   '.info.log')
path_log = "/mnt/data/log"
if not os.path.exists(path_log):
    os.makedirs(path_log)

errorFile = os.path.join(path_log, 'error.log')
infoFile  = os.path.join(path_log, 'info.log')
anaFile = os.path.join(path_log, 'ana.log')

def setup_logger(logger_name, log_file, level=logging.INFO):
    logDirPath = os.path.dirname(log_file)
    if not os.path.exists(logDirPath):
        os.makedirs(logDirPath)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # 创建一个handler，用于写入日志文件
    #fh = logging.FileHandler(log_file)
    #fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=50000000)
    #fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=50000000, backupCount=10)
    #fh = logging.handlers.TimedRotatingFileHandler(log_file, when='H',interval=12, backupCount=30)
    fh = logging.handlers.TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=7)
    fh.setLevel(level)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    #logger.addHandler(ch)
    return logger

error_name = 'error'
info_name  = 'info'
analisy_name = 'test'
lang_pair_name = ''
lang_src = ''
lang_target = ''
name = 'test post correct  error  info  nmt return'
nameArr = name.split()

def setLangPair(langPair):
    global lang_pair_name, lang_src, lang_target
    lang_pair_name = langPair
    langArr = lang_pair_name.split('2')
    if len(langArr) == 2:
        lang_src = langArr[0]
        lang_target = langArr[1]
    print(lang_pair_name, lang_src, lang_target)
    global path_log
    path_log = os.path.join(path_log, lang_pair_name)
    '''
    for name in nameArr:
        dirPath = os.path.join(path_log, lang_pair_name)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        namePath =  os.path.join(dirPath,   name + '.log')
        setup_logger(name, namePath)
    '''

#setup_logger(error_name, errorFile)
#setup_logger(info_name,  infoFile)
logQueue = Queue(10000)

def logToServer(message):
    global _isNeedSendLog
    if not _isNeedSendLog:
        return
    #key = "d9e23d93053f49ade2f8fce185acedd4"
    key = "9ae00f6ad40b6b1e65566c89ea98c41b"
    tag = "devops.apm.us"
    now = datetime.datetime.utcnow()
    now_date = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    now_date = str(int(time.time()))
    md5Str = str(hashlib.md5((tag+ ':' + now_date + ':' + key).encode("utf-8")).hexdigest())
    urlAppendPara = '?' +"tag=" + tag + '&timestamp=' + now_date + '&num=1' + '&signature=' +md5Str
    #print (urlAppendPara)
    url = "http://logagent.infra.funplus.net/log" + urlAppendPara
    r = requests.post(url, message.encode('utf-8'))
    #print (r.text)

def logToAnalysisServiceServer(message, logLevel='ERROR', fileName='', funcName=''):
    global lang_pair_name, lang_src, lang_target, logError
    global _isNeedSendLog
    if not _isNeedSendLog:
        return
    key = "wSPAlyil22HfUWtSnuuVwnzerLD3c4nX"
    tag = "alt.monitor"
    pid = "nmt"
    now = datetime.datetime.utcnow()
    now_date = str(int(time.time()))
    md5Str = str(hashlib.md5((tag+ ':'+ key + ':' + now_date ).encode("utf-8")).hexdigest())
    urlAppendPara = '?' + "tag=" + tag + '&timestamp=' + now_date + '&num=1' + '&signature=' +md5Str
    #print (urlAppendPara)
    #now.strftime('%Y-%m-%dT%H:%M:%SZ')
    #[2019-10-10 01:22:24,970]~[ERROR]~[2256(2430)]~[10.13.156.212@PushServer@operator()@PushTaskHelper.cpp:1387]~[]: log body
    dateStr = now.strftime('%Y-%m-%d %H:%M:%S,%f')
    ipAddress = get_host_ip()
    headArr = []
    headArr.append(dateStr)
    headArr.append(logLevel)
    headArr.append( '('+ str(os.getpid()) + ')')
    headArr.append(ipAddress + "@nmt_" + lang_src + "_" + lang_target + "@" + fileName + "@" + funcName)
    headArr.append('')
    bodyHeader = "[" + "]~[".join(headArr) + "]"
    url = "http://httpagent-nlp.ilivedata.com:11091/log" + urlAppendPara
    #print(url)
    body = bodyHeader + ":" + message
    _fpLog.write(logError, body)
    #r = requests.post(url, body.encode('utf-8'))
    #print (r.status_code, 'reslt:',  r.content)

class LogProcess(Thread):
    def run(self):
        while True:
            try:
                item = logQueue.get()
                if item and item.ltype == 'server':
                    logToServer(item.message)
                elif item and len(item.ltype) > 0:
                    logger = getLogger(item.ltype)
                    if logger:
                        logger.info(item.message)
                    send_to_log_server_file(item)
                if item.ltype == 'nmt' and '<unk> <unk> <unk> <unk> <unk>' in item.message:

                    logToAnalysisServiceServer(item.message, 'WARNING')
            except Exception as e:
                print(e)
                traceback.print_stack()


process = LogProcess()
process.start()

errorLog = logging.getLogger(error_name)
infoLog  = logging.getLogger(info_name)
anaLog =  logging.getLogger(analisy_name)

def getLogger(name):
    filepath = os.path.join(path_log,name + ".log")
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name, filepath)
    return logger

def log(modelName, message):
    needLog = False
    #if modelName in nameArr:
    if modelName :
        '''
        logger = getLogger(modelName)
        logger.info(message)
        '''
        if not logQueue.full():
            m = LogInfoClass()
            m.ltype = modelName
            m.message = message
            logQueue.put(m)

def get_parent_class_and_function_name():
    className = ''
    funcName = ''
    try:
        className = inspect.stack()[2][1]
        funcName = inspect.stack()[2][3]
    except Exception as e:
        print(e)
        print(traceback.format_stack())
    return className, funcName

def error(msg, name=''):
    try:
        errMsg = ''

        errMsg = errMsg + name
        errMsg = errMsg + str(msg)
        errMsg = errMsg + str(traceback.format_exc())
        errMsg = errMsg + str(traceback.format_stack())

        errorLog.error(errMsg)
        fileName, funcName = get_parent_class_and_function_name()
        logToAnalysisServiceServer(errMsg, 'ERROR', fileName, funcName)
    except Exception as e:
        print(e)

def test(msg, name=''):
    errMsg = ''
    if name:
        errMsg = errMsg + name
    errMsg = errMsg + msg
    anaLog.info(errMsg)

def info(msg, name=''):
    if not logQueue.full():
        m = LogInfoClass()
        m.ltype = 'info'
        m.message = msg
        logQueue.put(m)
        
# ***************** my *****************

class log_agent:
    global logError
    global logDebug
    global logKibana
    global _fpLog

    def __init__(self):
        self.log_dict = {}

    def getJsonStr(self):
        return json.dumps(self.__dict__)

    def write(self, log_time, lan, handel_name, ori_sen, cor_sen, message):
        self.log_dict["ipAddress"] = get_host_ip()
        self.log_dict["time"] = log_time
        self.log_dict["language"] = lan
        self.log_dict["moduleName"] = handel_name
        self.log_dict["sen_ori"] = ori_sen
        self.log_dict["sen_cor"] = cor_sen
        self.log_dict["result"] = message
        

        _fpLog.write(logKibana, json.dumps(self.log_dict, ensure_ascii=False))


def log_agent_write(log_time, lan, handel_name, ori_sen, cor_sen, message):
    log_agent_class = log_agent()
    log_agent_class.write(log_time, lan, handel_name, ori_sen, cor_sen, message)

if '__main__' == __name__:
    for i in range(10):
        info('main',  'info log')
        #analisy('analisy log')
