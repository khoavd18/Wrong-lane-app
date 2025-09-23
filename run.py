# run.py
# -*- coding: utf-8 -*-
import os, sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from wrong_lane.app import main

if __name__ == '__main__':
    main()
