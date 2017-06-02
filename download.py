#!/usr/bin/env python
#-*- coding:utf-8 -*-

import urllib
import sys

def download():

    url = sys.argv[1]
    title = sys.argv[2]
    urllib.urlretrieve(url,"{0}".format(title))

if __name__ == "__main__":
    download()
