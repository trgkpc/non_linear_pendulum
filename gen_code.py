#!/usr/bin/python3.6
import subprocess as sp

def run(cmd):
    print("cmd:",cmd)
    sp.run(cmd, shell=True)

target_file = "equation_of_state.py"

run(" ".join(["./autodiff.py",">",target_file]))
run(" ".join(["chmod","755",target_file]))
