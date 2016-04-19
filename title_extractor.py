#!/usr/bin/python3

def ExtractTitles(file):
    title_list = []
    f = open(file,encoding='latin_1')
    for line in f:
        if line[0]=='#':
            title = line[line.index('"')+1:line.rindex('"')]
            title_list.append(title)
    return title_list
        
if __name__=='__main__':
    import sys
    ExtractTitles(sys.argv[1])