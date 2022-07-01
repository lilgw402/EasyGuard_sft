# -*- coding: utf-8 -*-
'''
Created on Nov-13-20 17:09
stop_watch.py
@author: liuzhen.nlp
Description:
'''
import time


class StopWatch:
    """
    一个用于计时的类: 可以在某一段需要计时的代码开头调用start, 在结束调用elapsed来统计运行的时间.
    example:
        stop_watch = StopWatch()
        stop_watch.start()
        ...
        若干代码段
        ...
        elapsed = stop_watch.elapsed()
    """

    def __init__(self):
        self.start()

    def start(self) -> None:
        """
        获取当前的时间戳
        """
        self._startTime = time.time()

    def get_start(self) -> float:
        """
        返回start调用的时间
        """
        return self._startTime

    def elapsed(self, prec=3) -> float:
        """
        统计start开始到当前的时间
        """
        prec = 3 if prec is None or not isinstance(prec, (int)) else prec
        diff = time.time() - self._startTime
        return round(diff, prec)
