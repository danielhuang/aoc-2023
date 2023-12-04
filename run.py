import os
import sys
import subprocess

# Build the command to execute
command = ["cargo", "run", "--bin"] + sys.argv[1:]

# Use subprocess to execute the command
subprocess.run(command)
