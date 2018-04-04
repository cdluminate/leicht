#!/usr/bin/python3
'''
leichtBeautify -- Beautify the Ugly Unit Test Lines: Dash right!!
'''
import sys
import os

def beaut(filename):
    with open(filename) as f:
        lines = f.readlines()
    newlines = []
    for line in lines:
        if line.strip().startswith('//ut'):
            line = '//ut' + line.replace('//ut','').strip().rjust(75)
            newlines.append(line+'\n')
        elif line.strip().startswith('//>'):
            line = '//>' + line.replace('//>','').strip().rjust(76)
            newlines.append(line+'\n')
        else:
            newlines.append(line)
    with open(filename, 'w') as f:
        f.writelines(newlines)

if __name__=='__main__':
    try:
        beaut(sys.argv[1])
    except Exception as e:
        print("Error:", e)
