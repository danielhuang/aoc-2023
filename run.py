#!/usr/bin/env python

import os
import sys

os.execvp("cargo", ["cargo", "run", "--bin"] + sys.argv[1:])