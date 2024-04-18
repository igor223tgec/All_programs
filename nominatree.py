#! /usr/bin/python3

import subprocess
import os
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="""nominatree.py - For a phylogenetic tree 
  that has only sequence access codes, this program 
  uses the FASTA file that originated the phylogenetic 
  tree to name these elements. """)
  parser.add_argument('tree', help="Phylogenetic tree.")
  parser.add_argument('fasta', help="Fasta which the was made by.")
  args = parser.parse_args()

  codenames = {}
  codes = "".join(str(subprocess.check_output('grep ">" '+args.fasta+'|cut -f 2 -d ">"|cut -f 1 -d " "', shell=True))[1:].replace('"','').replace("'","").split("\n")).split('\\n')
  names = "".join(str(subprocess.check_output('grep ">" '+args.fasta+'|cut -f 2 -d "["|cut -f 1 -d "]"', shell=True))[1:].replace('"','').replace("'","").split('\n')).split('\\n')

  for i in range(0,len(codes)-1):
    if "(" in names[i] or ")" in names[i]:
      names[i]=names[i].replace("(","").replace(")","")
    codenames[codes[i]]=names[i]
  #  print(codes[i]+' - '+names[i])

  tree = open(args.tree, "r")
  tree = tree.readlines()[0]

  for n in codenames:
    tree = tree.replace(n, n+" - ["+codenames[n]+"]")
  #  print(n+" - "+codenames[n])

  with open('named_'+args.tree, 'w') as save:
    save.write(tree)

