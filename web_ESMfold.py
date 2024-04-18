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
parser.add_argument('-n')
parser.add_argument('-o')
parser.add_argument('-h', "--help", action='store_true')
args = parser.parse_args()

version = "1.0"

help = f"""
web_ESMfold - version {version} - 16 jan 2024

Usage: web_ESMfold.py -i <input_file> -o <output_directory>

Mandatory parameters:
-i input_file <file name>       Input file (FASTA sequence file)
-o output <directory name>      Output directory to store files

Additional parameters:
-n file_names* <id|all>         Names for labels:
                                  id - YP_003289293.1.pdb (default)
                                  all - YP_003289293.1_RNA-dependent_RNA_polymerase_Drosophila_totivirus_1.pdb

From example: YP_003289293.1 RNA-dependent RNA polymerase [Drosophila totivirus 1]
"""

def run_webESMfold(args):
  with open(args.i, 'r') as fasta:
    lines = fasta.readlines()

  sequences = {}
  for line in lines:
    if line.startswith(">"):
      if args.n == "id":
        new_seq = line[1:].split(" ", 1)[0]
      elif args.n == "all":
        new_seq = line[1:].replace(" ", "_")
        new_seq = new_seq.replace(":", "")
        new_seq = new_seq.replace("[", "").replace("]", "")
        new_seq = new_seq.replace("(", "").replace(")", "")
        new_seq = new_seq.replace("{", "").replace("}", "")
        new_seq = new_seq.replace("\n", "")

      else:
        print("Parameter -n not recognized. Choose between 'id' and 'all'.")
        sys.exit
    else:
      new_line = line.replace("\n", "").replace(" ", "").replace("-", "")
      if new_seq in sequences:
        old_line = sequences[new_seq]
        sequences[new_seq] = old_line+new_line
      else:
        sequences[new_seq] = new_line

  for seq in sequences:
    outpath = os.path.realpath(args.o)
    outfile = seq+".pdb"
    if os.path.isfile(f"{outpath}/{seq}.pdb"):
      print(f"{outfile} already exists. Skipping...")
      continue
    cmd_ESMfold = f'curl -X POST --data "{sequences[seq]}" https://api.esmatlas.com/foldSequence/v1/pdb/ -k > {outpath}/{outfile}'
    try:
      subprocess.call(cmd_ESMfold, shell = True)
    except Exception as error:
      print("An exception occurred:", error)
    else:
      print(f"Generating: {outfile}")
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
Use the command template below adding any optional parameters you want.

web_ESMfold -i [input_file] -o [output]""")
    sys.exit()

def check_fasta(fasta):
  if os.path.isfile:
    with open(fasta, 'r') as file:
      lines = file.readlines()
    if not lines:
      print("Error: Input file is empty.")
      sys.exit()
    if not lines[0].startswith(">"):
      print("Error: Input file is not in FASTA format.")
      sys.exit()
  else:
    print("Error: Input file not found in the working directory.")
    sys.exit()
  type_header = "ncbi"
  for line in lines:
    if line.startswith(">") and not re.match(r'>.* .* \[.*\]', line):
      type_header= "not_ncbi"
      break

if __name__ == '__main__':
  if not len(sys.argv)>1:
    print(help)
  elif args.help == True:
    print(help)
  else:
    if args.n == None:
      args.n = "id"
    mandatory_param_check(args)
    check_fasta(args.i)
    if os.path.isdir(args.o) == False:
      os.mkdir(args.o)
    run_webESMfold(args)
