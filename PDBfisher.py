#!/usr/bin/env python3

from multiprocessing import cpu_count
import os
import argparse
import subprocess
import sys
import re
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-i')
parser.add_argument('-o')
parser.add_argument('-h', "--help", action='store_true')
parser.add_argument('-v', "--version", action='store_true')
args = parser.parse_args()

version = "1.0"

help = f"""
PDBfisher - version {version} - 16 jan 2024

Usage: PDBfisher.py -i <input_file> -o <output_directory>

Parameters:
-i input_file <file name>       Input file (List of sequences to obtain)
-o output <directory name>      Output directory to store files
-h help                         This page of help
-v version			Prints actual version of this program
"""

def fishing(args):
  with open(args.i, 'r') as codes:
    lines = codes.readlines()

  end = len(lines)

  i = 0
  for line in lines:
    i += 1

    new_seq = line.replace(" ", "_")
    new_seq = new_seq.replace(";", "")
    new_seq = new_seq.replace("[", "").replace("]", "")
    new_seq = new_seq.replace("(", "").replace(")", "")
    new_seq = new_seq.replace("{", "").replace("}", "")
    new_seq = new_seq.replace("\n", "")

    outpath = os.path.realpath(args.o)
    outfile = new_seq+".pdb"
    if os.path.isfile(f"{outpath}/{new_seq}.pdb"):
      print(f"{outfile} already exists. Skipping... ({str(i)}/{str(end)})")
      continue
    cmd_ESMfold = f'curl https://alphafold.ebi.ac.uk/files/{new_seq}-model_v4.pdb > {outpath}/{outfile}'
    try:
      subprocess.call(cmd_ESMfold, shell = True)
    except Exception as error:
      print("An exception occurred:", error)
    else:
      print(f"Generating: {outfile}... {str(i)}/{str(end)}")
    time.sleep(3)

def mandatory_param_check(args):
  not_specified = []
  if args.i == None:
    not_specified.append("-i [input_file]")
  if args.o == None:
    not_specified.append("-o [output]")

  if len(not_specified) != 0:
    if len(not_specified) == 1:
      not_specified = not_specified[0]
    else:
      not_specified = ", ".join(not_specified)
    print("""Error: The command is missing mandatory parameters: """+not_specified+""".
Use the command template below:

PDBfisher -i [input_file] -o [output]""")
    sys.exit()

if __name__ == '__main__':
  if not len(sys.argv)>1:
    print(help)
  elif args.help == True:
    print(help)
  elif args.version == True:
    print(f"\nPDBfisher - version {version}\n")
  else:
    mandatory_param_check(args)
    if os.path.isdir(args.o) == False:
      os.mkdir(args.o)
    fishing(args)
