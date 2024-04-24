#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import struct
import threading
import socket
if sys.version > '3':
    import queue
else:
    import Queue as queue

MAX_LOG_QUEUE_SIZE = 10000

class FPLog(object):
    def __init__(
            self,
            host='/tmp/fplog.sock',
            pid=0,
            timeout=3,
            conn_pool=1
    ):
        self.stop = False
        self.queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.flush_condition = threading.Condition()
        self.host = host
        self.pid = pid
        self.timeout = timeout
        self.conn_pool = conn_pool
        self.flush_threads = set()
        for _ in range(self.conn_pool):
            t = threading.Thread(target=FPLog.flush, args=(self,))
            t.setDaemon(True)
            t.start()
            self.flush_threads.add(t)

    def _connect(self):
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(self.timeout)
            conn.connect(self.host)
            conn.settimeout(None)
            return conn
        except:
            return None

    def _close(self, conn):
        try:
            if conn != None:
                conn.close()
        except:
            pass

    def _send(self, conn, data):
        try:
            conn.sendall(data)
            return True
        except:
            return False

    def _read(self, conn):
        try:
            data = conn.recv(1)
            if not data or data != b'1':
                return False
            return True
        except:
            return False

    def flush(self):
        conn = self._connect()
        while True:
            if self.stop:
                break
            with self.flush_condition:
                self.flush_condition.wait()

            buffer = []
            with self.queue_lock:
                while True:
                    try:
                        buffer.append(self.queue.get(block = False))
                    except:
                        break
            if len(buffer) == 0:
                continue

            if conn == None:
                conn = self._connect()

            if conn == None:
                with self.queue_lock:
                    for data in buffer:
                        self.queue.put(data)
                continue

            for data in buffer:
                if not (self._send(conn, data) and self._read(conn)):
                    self._close(conn)
                    conn = self._connect()
                    self._send(conn, data)
                    self._read(conn)
            
        self._close(conn)

    def write(self, tag, data):
        payload = struct.pack('=B', len(str.encode(tag))) + str.encode(tag) + struct.pack('H', 0) + struct.pack('=L', self.pid) + struct.pack('=L', len(str.encode(data))) + str.encode(data)
        with self.queue_lock:
            if self.queue.qsize() > MAX_LOG_QUEUE_SIZE:
                raise RuntimeError('Log Queue reach max size limited, log will be droped')
            self.queue.put(payload)
        with self.flush_condition:
            self.flush_condition.notify()

    def forcewrite(self, tag, data):
        try:
            self.write(tag, data)
        except:
            pass

    def destory(self):
        self.stop = True

if __name__ == '__main__':
    l = FPLog('/tmp/fplog.sock', conn_pool=3)
    l.forcewrite('test.file', 'this is log body')
    l.write('test.file', 'this is log body')

    import time
    time.sleep(1)