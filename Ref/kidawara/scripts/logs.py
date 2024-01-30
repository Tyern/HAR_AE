#! -*- coding: utf-8
import logging
import os
import os.path as path
import sys

__all__ = ["loglevels", "config_logger"]

loglevels = ["TRACE0", "TRACE1", "TRACE2", "TRACE3", "TRACE4",
             "TRACE5", "TRACE6", "TRACE7", "TRACE8", "TRACE9",
             "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

##### Global logger settings #####
for i in range(0, 10):  # 独自追加したログレベルの名称をloggingパッケージに登録
    name = loglevels[i]
    logging.addLevelName(i, name)
    setattr(logging, name, i)  # ログ名を設定
##################################


def config_logger(logformat: str, loglevel: str = "INFO",  logfile: str = None):
    if logfile is None:
        logging.basicConfig(level=loglevel, format=logformat,
                            stream=sys.stderr)  # ログファイル未指定の場合、標準エラーに出力
    else:
        logdir = path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
        logging.basicConfig(level=loglevel, format=logformat,
                            filename=logfile)  # ログファイル未指定の場合、標準エラーに出力
