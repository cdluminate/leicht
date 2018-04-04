#!/usr/bin/python3
'''
leichtUT -- Unit Test for Leicht
'''
import sys
import os

uttemplate_head = '''
#include "leicht.hpp"
#include <cblas-openblas.h>

string _msg_;
#define TS(msg) do { \
  _msg_ = msg; \
  cerr << endl << "\\x1b[32;1m... " << _msg_ << " [ .. ]\\x1b[m"; \
 } while (0)
#define TE do { \
  for (long i = (long)_msg_.size()+11; i > 0; i--) cerr << "\b"; \
  cerr << "\\x1b[32;1m>>> " << _msg_ << " [ OK ]\\x1b[m" << endl; \
 } while (0)

int
main(void)
{
        cout << ":::[..] Unit Tests" << endl;
'''
uttemplate_tail = '''
        cout << ":::[OK] Unit Tests" << endl;
        return 0;
}
'''

def genut(filename):
    '''
    generate unit test from the comments of a given file
    '''
    utname = []
    utcode = []
    with open(filename) as f:
        lines = f.readlines()
    # preprocess pass: filter
    lines = [ line.strip() for line in lines ]
    lines = [ line for line in lines if line.startswith("//ut")
              or line.startswith("//>") ]
    # first pass: collect tests
    for line in lines:
        if line.startswith('//ut'):
            utname.append(line.replace('//ut','').strip())
            utcode.append([])
        else:
            utcode[-1].append(line.replace('//>','').strip())
    # second pass: write test file
    with open(filename+'_ut.cc', 'w') as f:
        f.write(uttemplate_head);
        for i in range(len(utname)):
            f.write("cerr << \"{}\";".format("({}/{})".format(i+1,len(utname)).rjust(87)))
            f.write("""
TS("#NAME#"); {
#CODE#
}; TE;
            """.replace('#NAME#', utname[i].rjust(76)).replace(
                '#CODE#', '\n'.join(utcode[i])))
        f.write(uttemplate_tail)

if __name__=='__main__':
    try:
        genut(sys.argv[1])
    except Exception as e:
        print("Error:", e)
