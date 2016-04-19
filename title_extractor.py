#!/usr/bin/python3
import pdb

def ExtractTitles(file):
    title_list = []
    f = open(file,encoding='latin_1')
    for line in f:
        if line[0]=='#':
            try:
                title = line[2:line.rindex('(')]
                title = title.rstrip('" ').lstrip('''#" ''').lower()
            except:
                pdb.set_trace()
            title_list.append(title)
    return title_list
        
if __name__=='__main__':
    import sys
    ExtractTitles(sys.argv[1])