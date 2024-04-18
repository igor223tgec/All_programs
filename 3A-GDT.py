#!/usr/bin/env python3

from multiprocessing import cpu_count
from shutil import copyfile
import re
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)


version = "1.30"

num_threads = cpu_count()

help = """
3A-DGT - Triple-A DGT version """+version+""" - 31 mar 2024
The All-Against-All Distance Graphical Tool
(c) 2023. Igor Custódio dos Santos & Arthur Gruber

Usage: 3A-DGT -i <input_file> -method <method> -o <output_directory>

-i input <file name>       	Input file (FASTA sequence file)
-s structures <directory name>	Structures directory (PDBs directory)
-p protocol <a|l|s|3|2> (default=all methods)
                                     a - pairwise aa sequence alignment
                                     l - maximum likelihood
				     s - pairwise structural alignment
				     3 - pairwise 3di character alignment
				     2 - combined alignment of 3di and amino acid sequences
                                     Examples: -m als32 -m al

Additional parameters:
-c color                        Color palette from matplotlib package for graphs (default=RdBu)
-f mafft <parameters>		Only "--maxiterate" and "--thread" MAFFT parameters can be change.
				e.g. -mafft "--maxiterate 1000 --thread 20"
-h help                       	This help page
-l lines <yes|no>               Use cell dividing lines on heatmap (default=yes)
-n names* <id|taxon|all>        Names for labels:
                                  id - YP_003289293.1 (default)
                                  name - Drosophila totivirus 1
                                  all - YP_003289293.1 - Drosophila totivirus 1
-o output <directory name>      Output directory to store files
-q iqtree <parameters>		Only "-m", "-bb" and "-nt" IQTREE parameters can be changed.
				e.g. -iqtree "-m Q.pfam+F+R6 -bb 1000 -nt 20"
-v version			Show the program's version.

*Only valid for NCBI naming format:
YP_003289293.1 RNA-dependent RNA polymerase [Drosophila totivirus 1]
For other naming formats, all terms are used
"""

parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
parser.add_argument('-i')
parser.add_argument('-s')
parser.add_argument('-p', default='als23')
parser.add_argument('-o')
parser.add_argument('-c', default='RdBu')
parser.add_argument('-l', default='yes')
parser.add_argument('-n', default="id")
parser.add_argument('-h', '--help', action='store_true')
parser.add_argument('-v', '--version', action='store_true')
parser.add_argument('-f', default='--maxiterate 1000 --thread '+str(round(num_threads/2)))
parser.add_argument('-q', default='-m Q.pfam+F+R6 -bb 1000 -nt '+str(round(num_threads/2)))
args = parser.parse_args()

def mandatory_param_check(args):
  not_specified = []
  if args.i == None:
    not_specified.append("-i [input_file]")
  if args.p == None:
    not_specified.append("-method [method]")
  if args.o == None:
    not_specified.append("-o [output]")

  if len(not_specified) != 0:
    if len(not_specified) == 1:
      not_specified = not_specified[0]
    else:
      not_specified = ", ".join(not_specified)
    print("""Error: The command is missing mandatory parameters: """+not_specified+""".
Use the command template below adding any optional parameters you want.

3A-GDT -i [input_file] -method [method] -o [output]""")
    sys.exit()


def check_fasta(fasta):
  if fasta == None:
    return "ncbi", "protein"
  else:
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

    is_protein = None  # Inicializamos como None para determinar o tipo na primeira sequência.
    for line in lines:
      line = line.strip()
      if line.startswith(">"):
        # Se a linha começa com ">", é um cabeçalho e não contém sequências.
        continue
      elif any(base in "RDEILKMrdeilkm" for base in line.upper()):
        is_protein = True
        break
      else:
        # Se não encontramos caracteres não nucleotídicos até agora, assumimos
        # que a sequência é de nucleotídeos.
        is_protein = False

    if is_protein:
      return type_header, "protein"
    elif is_protein is not None:
      return type_header, "nucleotide"

def check_pdb(pdb_dict):
  if pdb_dict != None:
    if os.path.isdir(pdb_dict) == False:
      print(f"\nERROR: '{pdb_dict}' (-p) is not a directory.")
      sys.exit()

    file_list = os.listdir(pdb_dict)

    if len(file_list) == 0:
      print(f"\nERROR: The directory '{pdb_dict}' (-p) is empty.")
      sys.exit()

    for pdb in file_list:
      if pdb.endswith(".pdb")==False:
        print(f'In "{pdb_dict}" directory, the file "{pdb}" might not be a pdb format file.\nPlease take it off from the input directory or rename it with pdb extention.')
        sys.exit()

def check_output_dir(output):
  if output != None:
    return output
  else:
    i = 1
    while True:
      if os.path.isdir(f"output{str(i)}"):
        i +=1
        continue
      else:
        save_output_dir = f"output{str(i)}"
        break
    print("Output directory not specified. Saving files in {save_output_dir}")

  return save_output_dir


def check_param(args):
  if args.l not in ['yes', 'no']:
    print("Error: Line parameter (-l) not recognized.")
    sys.exit()

  color = args.c
  if not sns.color_palette(color):
    print("Error: Color parameter (-c) not recognized.")
    sys.exit()

  if args.n not in ['id', 'name', 'all']:
    print("Error: Label parameter (-n) not recognized.")
    sys.exit()

def isInt(value):
  try:
    int(value)
    return True
  except:
    return False

def check_mafft(param_mafft):
  num_threads = cpu_count()
  mafft_list = param_mafft.split(' ')
  if len(mafft_list) % 2 != 0:
    print("ERROR: MAFFT parameters not accepted. It has an odd number of parameters.")
    sys.exit()
  for i in range(0, len(mafft_list), 2):
    if mafft_list[i] == "--maxiterate":
      if not isInt(mafft_list[i+1]):
        print("ERROR: MAFFT parameter --maxiterate is not integer")
        sys.exit()
    elif mafft_list[i] == "--thread":
      if not isInt(mafft_list[i+1]):
        print("ERROR: MAFFT parameter --thread is not integer")
        sys.exit()
      else:
        if int(mafft_list[i+1]) > num_threads:
          print("Number of processors ("+mafft_list[i+1]+") exceeds the total CPUs available on the server. Using "+str(round(num_threads))+" CPUs.")
          mafft_list[i+1] = str(round(num_threads))
    else:
      print("ERROR: MAFFT parameter '"+mafft_list[i]+"' not accepted by 3A-DGT. Please correct your syntax.")
      sys.exit()
  if len(mafft_list) == 2:
    if "--maxiterate" not in mafft_list:
      mafft_list.append("--maxiterate")
      mafft_list.append("1000")
    if "--thread" not in mafft_list:
      mafft_list.append("--thread")
      mafft_list.append(str(round(num_threads/2)))

  mafft_cmd = " ".join(mafft_list)

  return mafft_cmd


def check_iqtree(param_iqtree):
  num_threads = cpu_count()
  iqtree_list = param_iqtree.split(' ')
  if len(iqtree_list) % 2 != 0:
    print("ERROR: IQ-TREE parameters not accepted. It has an odd number of parameters.")
    sys.exit()
  for i in range(0, len(iqtree_list), 2):
    if iqtree_list[i] == "-bb":
      if not isInt(iqtree_list[i+1]):
        print("ERROR: IQ-TREE parameter -bb is not integer.")
        sys.exit()
      if isInt(iqtree_list[i+1]) and int(iqtree_list[i+1]) < 1000:
        print("ERROR: IQ-TREE parameter -bb must be greater than 1000.")
        sys.exit()
    elif iqtree_list[i] == "-nt":
      if not isInt(iqtree_list[i+1]):
        print("ERROR: IQ-TREE parameter -nt is not integer")
        sys.exit()
      else:
        if int(iqtree_list[i+1]) > num_threads:
          print("Number of processors ("+iqtree_list[i+1]+") exceeds the total CPUs available on the server. Using "+str(round(num_threads))+" CPUs.")
          iqtree_list[i+1] = str(round(num_threads))
    elif iqtree_list[i] == "-m":
      continue
    else:
      print("ERROR: IQ-TREE parameter '"+iqtree_list[i]+"' not accepted by 3A-DGT. Please correct your syntax.")
      sys.exit()

  if len(iqtree_list) == 2:
    if "-bb" not in iqtree_list:
      iqtree_list.append("-bb")
      iqtree_list.append("1000")
    if "-nt" not in iqtree_list:
      iqtree_list.append("-nt")
      iqtree_list.append(str(round(num_threads/2)))
    if "-m" not in iqtree_list:
      iqtree_list.append("-m")
      iqtree_list.append("Q.pfam+F+R6")

  iqtree_cmd = " ".join(iqtree_list)
  return iqtree_cmd

def log_header(log, args, type_sequence, type_header, version):
  now = datetime.datetime.now()
  date_format = "%m-%d-%Y %H:%M:%S"
  formatted_date = now.strftime(date_format)

  if args.i != None:
    fasta_name = args.i

    count_fasta = 0
    with open(args.i, 'r') as fasta:
      line = fasta.readline()
      while line:
        if line.startswith('>'):
          count_fasta += 1
        line = fasta.readline()

    if type_sequence == "protein":
      log_type_sequence = "Protein"
    elif type_sequence == "nucleotide":
      log_type_sequence = "Nucleotide"

    if type_header == "ncbi":
      log_type_header = "NCBI standard"
    elif type_header == "not_ncbi":
      log_type_header = "Not NCBI standard"

  else:
    fasta_name = "NOT SPECIFIED"
    count_fasta = " - "
    log_type_sequence = " - "
    log_type_header = " - "


  if args.s != None:
    pdb_name = args.s
    count_pdb = len(os.listdir(args.s))
  else:
    pdb_name = "NOT SPECIFIED"
    count_pdb = " - "

  method_dict = {"a": "(a) pairwise aa sequence alignment", "l": "(l) maximum likelihood", "s": "(s) pairwise structural alignment", "3": "(3) pairwise 3di character alignment", "2": "(2) combined alignment of 3di and amino acid sequences"}
  methods = ""
  for met in method_dict:
    if met in args.p:
      methods += f"\n  - {method_dict[met]}"


  header_dict ={"id": "Identification code", "taxon": "Taxon", "all": "Identification code and taxon"}
  if args.n in header_dict:
    log_names = f"{header_dict[args.n]} ({args.n})"

  if args.f == '--maxiterate 1000 --thread '+str(round(num_threads/2)):
    log_mafft = args.f+" (default)"
  else:
    log_mafft = args.f

  if args.q == '-m Q.pfam+F+R6 -bb 1000 -nt '+str(round(num_threads/2)):
    log_iqtree = args.q+" (default)"
  else:
    log_iqtree = args.q

  log.write(f"""3A-DGT - Triple-A DGT version {version} - 08 jan 2024
The All-Against-All Distance Graphic Tool
(c) 2023. Igor Custódio dos Santos & Arthur Gruber

***Logfile (Date: {formatted_date})***

Mandatory parameters:

-Input fasta file: {fasta_name}
  - Number of sequences: {count_fasta}
  - Type of sequences: {log_type_sequence}
  - Type of headers: {log_type_header}
-Input PDB directory: {pdb_name}
  - Number of files: {count_pdb}
- Method: {methods}
- Output directory: {args.o}

Optional parameters:

- Heatmaps color: {args.c}
- Names in heatmaps: {log_names}
- Heatmap lines: {args.l}
- Mafft parameters: {log_mafft}
- IQ-TREE parameters: {log_iqtree}
""")

def where_to_save_graphics(output):
  if os.path.isdir(output):
    i = 2
    print(f"Output directory alrealdy exists: {output}")
    if os.path.isdir(f"{output}/graphics_dir"):
      while True:
        if os.path.isdir(f"{output}/graphics_dir_{str(i)}"):
          i +=1
          continue
        else:
          save_graphics_dir = f"{output}/graphics_dir_{str(i)}"
          logfile = f"{output}/{output}_{str(i)}.log"
          break
    else:
      save_graphics_dir = f"{output}/graphics_dir"
      logfile = f"{output}/{output}.log"
  else:
    os.mkdir(output)
    save_graphics_dir = f"{output}/graphics_dir"
    logfile = f"{output}/{output}.log"

  return save_graphics_dir, logfile

def entry_file(args, type_header):
  if type_header == 'not_ncbi':
    with open(args.i, 'r') as file:
      with open(f"{args.o}/renamed_{args.i}", 'w') as file2:
        for line in file:
          if line.startswith(">"):
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace(']', '')
            line = line.replace('[', '')
            line = line.replace(' ', "_")
          file2.write(line)
    enter = f"{args.o}/renamed_{args.i}"

  elif type_header == "ncbi":
    enter = args.i

  return enter

def correct_label(data, args):
  if args.n == 'id':
    return data
  elif args.n == "name" or args.n == "all":
    new_data = []
    with open(args.i, "r") as read_fasta:
      for line in read_fasta:
        if line.startswith(">"):
          new_data.append([line[1:].split(' ')[0], line.split("[")[1].rsplit("]")[0]])

    for o in data:
      for t in o:
        for n in new_data:
          if t == n[0]:
            if args.n == "name":
              data[data.index(o)][o.index(t)] = n[1]
            elif args.n == "all":
              data[data.index(o)][o.index(t)] = n[0]+" - "+n[1]
    return data

def run_needle(fasta_file, output_dir, type_header, type_sequence, log):
  save_needle_dir = f"{output_dir}/needle_dir"
  log.write(f"\nSaving in directory: {output_dir}/needle_dir\n")
  if os.path.isdir(save_needle_dir) == False:
    os.mkdir(save_needle_dir)
  sequencias = []  # Inicializa uma lista para armazenar as sequências
  sequencia_atual = []  # Inicializa uma lista temporária para armazenar a sequência atual

  with open(fasta_file, 'r') as arquivo:
    for linha in arquivo:
      linha = linha.strip()
      if linha.startswith(">"):
        # Se a linha é um cabeçalho, isso indica o início de uma nova sequência
        if sequencia_atual:
          sequencias.append(sequencia_atual)  # Adiciona a sequência à lista
        sequencia_atual = [linha]  # Inicia uma nova sequência com o cabeçalho
      else:
        # Se a linha não é um cabeçalho, ela faz parte da sequência atual
        sequencia_atual.append(linha)

    if sequencia_atual:
      sequencias.append(sequencia_atual)  # Adiciona a última sequência


  needle_file = save_needle_dir+"/"+output_dir+".needle"
  stdout_needle_file = save_needle_dir+"/"+output_dir+"_stdout.txt"
  with open(needle_file, 'w') as open_needle:
    open_needle.write("")
  with open(stdout_needle_file, 'w') as open_needle:
    open_needle.write("")


  for seq in sequencias:
    in_seq = seq[0][1:].split(' ')[0]
    in_seq_file = seq[0][1:].split(' ')[0]+'.fasta'
    out_seq_file = seq[0][1:].split(' ')[0]+'.needle'
    with open(in_seq_file, 'w') as arquivo:
      arquivo.write('\n'.join(seq))
    with open('comparated_sequences.fasta', 'w') as arquivo:
      for l in sequencias[sequencias.index(seq):]:
        arquivo.write('\n'.join(l)+'\n')
    try:
      if type_sequence == 'protein':
        subprocess.call(f"needle -sprotein1 {in_seq_file} -sprotein2 comparated_sequences.fasta -gapopen 10.0 -gapextend 0.5 -datafile EBLOSUM62 -outfile {out_seq_file} >{stdout_needle_file}", shell = True)
      if type_sequence == 'nucleotide':
        subprocess.call(f"needle -snucleotide1 {in_seq_file} -snucleotide2 comparated_sequences.fasta -gapopen 10.0 -gapextend 0.5 -datafile EDNAFULL -outfile out_seq_file >{stdout_needle_file}", shell = True)
    except Exception as erro_needle:
      log.write(f"\nError at running Needle:\n{erro_needle}")
      print(f"Error at running Needle:\n{erro_needle}")
      sys.exit
    os.remove(in_seq_file)
    os.remove('comparated_sequences.fasta')
    with open(out_seq_file, 'r') as open_moment_needle:
      with open(needle_file, 'a') as open_needle:
        for linha in open_moment_needle:
          open_needle.write(linha)
    os.remove(out_seq_file)

def data_from_needle(file, type_sequence):
  data = []
  data_ident = []
  data_sim = []
  with open(file, 'r') as open_needle:
    pair_sim = []
    pair_ident = []
    for needle_line in open_needle:
      if needle_line.startswith("# 1:"):
        pair_sim.append(needle_line[5:-1])
        pair_ident.append(needle_line[5:-1])
      elif needle_line.startswith("# 2:"):
        pair_sim.append(needle_line[5:-1])
        pair_ident.append(needle_line[5:-1])
      elif needle_line.startswith("# Identity:"):
        pair_ident.append(round(float(needle_line.split("(")[1].split("%")[0]), 2))
      if type_sequence == "protein":
        if needle_line.startswith("# Similarity:"):
          pair_sim.append(round(float(needle_line.split("(")[1].split("%")[0]), 2))
      else:
        continue
      if len(pair_sim) == 3:
        data_sim.append(pair_sim)
        pair_sim = []
      if len(pair_ident) == 3:
        data_ident.append(pair_ident)
        pair_ident = []
    data.append(data_ident)
    if type_sequence == "protein":
      data.append(data_sim)
  return data

def save_matrix(perc_list, save_path, log):
  matrix_name = save_path

  if save_path.endswith("_ident_matrix.csv"):
    matrix_title_log = "Identity"
  elif save_path.endswith("_simil_matrix.csv"):
    matrix_title_log = "Similarity"
  elif save_path.endswith('_tmscores_matrix.csv'):
    matrix_title_log = "TM-scores"
  elif save_path.endswith('_3Di_simil_matrix.csv'):
    matrix_title_log = "3Di Characters Similarity"
  elif save_path.endswith('_2.5Di_simil_matrix.csv'):
    matrix_title_log = "2.5Di Characters Scores"


  order = []
  for l in perc_list:
    if l[0] not in order:
      order.append(l[0])
  matriz = []
  for k in range(len(order)):
    linha = [None] * len(order)
    matriz.append(linha)

  for element in perc_list:
    i = order.index(element[0])
    j = order.index(element[1])
    if i < j:
      h = i
      i = j
      j = h
    matriz[j][i] = matriz[i][j] = element[2]

  table = {}
  table[''] = order
  for l in range(0, len(matriz)):
    table[order[l]] = matriz[l]
  final_table = pd.DataFrame(table)
  final_table.set_index('', inplace=True)
  final_csv = final_table.to_csv(index = True)
  with open(matrix_name, "w") as file:
    file.write(final_csv)

  log.write(f"""
{matrix_title_log} matrix in csv format ({matrix_name}):

{final_csv}

""")

def run_mafft(fasta_file, output, cmd_mafft, log):
  save_align_dir = f"{output}/mafft_dir"
  log.write(f"\nSaving in directory: {save_align_dir}\n")
  if os.path.isdir(save_align_dir) == False:
    os.mkdir(save_align_dir)
  real_path_fasta = os.path.realpath(fasta_file)
  real_path_output = os.path.realpath(save_align_dir+"/"+output+".align")
  error_out = os.path.realpath(save_align_dir+"/error_"+output+"_mafft")

  cmd = 'mafft '+cmd_mafft+' '+real_path_fasta+' > '+real_path_output+' 2> '+error_out
  log.write(f"\nMAFFT command: {cmd}\n")
  subprocess.call(cmd, shell=True)

def run_iqtree(align_file, output, cmd_iqtree, log):
  save_iqtree_dir = f"{output}/iqtree_dir"
  log.write(f"\nSaving in directory: {save_iqtree_dir}\n")
  if os.path.isdir(save_iqtree_dir) == False:
    os.mkdir(save_iqtree_dir)
  copyfile(align_file, save_iqtree_dir+"/"+output)
  real_path_align = os.path.realpath(f"{save_iqtree_dir}/{output}")
  error_out = os.path.realpath(f"{save_iqtree_dir}/error1_{output}_iqtree")
  error_out2 = os.path.realpath(f"{save_iqtree_dir}/error2_{output}_iqtree")
  cmd = f"iqtree2 -s {real_path_align} {cmd_iqtree} 1>{error_out} 2>{error_out2}"
  log.write(f"\nIQ-TREE command: {cmd}\n")
  try:
    subprocess.call(cmd, shell=True)
  except Exception as err:
    print("IQTREE reported an error:\n\n"+str(err))
  else:
    print("Running IQ-TREE...")
  os.remove(save_iqtree_dir+"/"+output)
  if os.path.isfile(f"{save_iqtree_dir}/{output}.mldist") == False:
    print("Parameter -m of IQTREE is not valid. Please correct your sitaxe.")
    sys.exit()

def data_from_mldist(mldist_file, log):
  data = []
  log.write("\nMaximum likelihood distance matrix:\n\n")
  with open(mldist_file, 'r') as arquivo:
    lines = arquivo.readlines()
    for line in lines[1:]:
      log.write(line)
      for distance in line.split()[1:]:
        data.append([line.split()[0], lines[line.split()[1:].index(distance)+1].split()[0], round(float(distance), 4)])
  log.write("\n")
  return data

def run_tmalign(input_dir, output, log):
  output_tmalign = f"{output}/TMalign_dir"
  log.write("\nSaving in directory: output_tmalign\n")
  if os.path.isdir(output_tmalign) == False:
    os.mkdir(output_tmalign)

  file_list = os.listdir(input_dir)

  for pdb in file_list:
    if pdb.endswith(".pdb")==False:
      print(f'In "{input_dir}" directory, the file "{pdb}" might not be a pdb format file.\nPlease take it off from the input directory or rename it with pdb extention.')
      sys.exit()

  combinations = []
  for yfile in file_list:
    for xfile in file_list:
      if xfile==yfile:
        continue
      if [yfile, xfile] in combinations or [xfile, yfile] in combinations:
        continue
      combinations.append([yfile, xfile])

  log.write("\nTM-align command model: TMalign pdb_file1 pdb_file2 -o directory_output -a T 1>tmalign.log\n")

  for pair in combinations:
    out1 = pair[0][:-4]
    out2 = pair[1][:-4]

    if ":" in pair[0]:
      out1 = out1.replace(":", "")
    if " " in pair[0]:
      out1 = out1.replace(" ", "_")
    if ":" in pair[1]:
      out2 = out2.replace(":", "")
    if " " in pair[1]:
      out2 = out2.replace(" ", "_")
    if os.path.isdir(f"{output_tmalign}/{out1}_x_{out2}")==False and os.path.isdir(f"{output_tmalign}/{out2}_x_{out1}")==False:
      os.mkdir(f"{output_tmalign}/{out1}_x_{out2}")
    elif os.path.isdir(f"{output_tmalign}/{out1}_x_{out2}") and os.path.isfile(f"{output_tmalign}/{out1}_x_{out2}/{out1}_x_{out2}.log"):
      continue
    elif os.path.isdir(f"{output_tmalign}/{out2}_x_{out1}") and os.path.isfile(f"{output_tmalign}/{out2}_x_{out1}/{out2}_x_{out1}.log"):
      continue

    tmalign_cmd = f"TMalign {input_dir}/'{pair[0]}' {input_dir}/'{pair[1]}' "
    tmalign_cmd += f"-o {output_tmalign}/{out1}_x_{out2}/{out1}_x_{out2} "
    tmalign_cmd += f"-a T 1>{output_tmalign}/{out1}_x_{out2}/{out1}_x_{out2}.log"
    print(tmalign_cmd)


    try:
      subprocess.call(tmalign_cmd, shell=True)
    except:
      print(f"Error at running TM-align with {pair[0]} vs {pair[1]}.")
      sys.exit()

def data_from_tmalign(output):
  data = []
  all = []
  for tmalign_dirs in os.listdir(f"{output}/TMalign_dir"):
    if "_x_" not in tmalign_dirs:
      continue
    if tmalign_dirs.split("_x_")[0] not in all:
      all.append(tmalign_dirs.split("_x_")[0])
    if tmalign_dirs.split("_x_")[1] not in all:
      all.append(tmalign_dirs.split("_x_")[1])
    with open(f"{output}/TMalign_dir/{tmalign_dirs}/{tmalign_dirs}.log", 'r') as open_moment_tmalign:
      for line in open_moment_tmalign:
        if "if normalized by average length of two structures" not in line:
          continue
        data.append([tmalign_dirs.split("_x_")[0], tmalign_dirs.split("_x_")[1], float(line.split("score= ")[1].split(" (if")[0])])
#  print(all)
 # print(len(all))
  for same in all:
    if [same, same, 1.0] not in data:
      data.append([same, same, 1.0]) 
  return data

def run_foldseek(args, log):
  if args.s != None and args.s.endswith("/"):
    args.s = args.s[:-1]
  if os.path.isdir(f"{args.o}/foldseek_dir") == False:
    os.mkdir(f"{args.o}/foldseek_dir")
  log.write(f"Saving in: {args.o}/foldseek_dir")
  if args.s != None and args.s.endswith("/"):
    entry_pdbs = f'{args.s}*'
  else:
    entry_pdbs = f'{args.s}/*'
  num_threads = str(round(cpu_count()/2))
  temp_out = f'{args.o}_foldseek_output'
  erro1 = f'{args.o}/foldseek_dir/erro1'
  erro2 = f'{args.o}/foldseek_dir/erro2'
  try:
    log.write(f'\nfoldseek structureto3didescriptor -v 0 --threads {num_threads} --chain-name-mode 1 {entry_pdbs} {temp_out} >{erro1} 2>{erro2}\n')
    subprocess.call(f'foldseek structureto3didescriptor -v 0 --threads {num_threads} --chain-name-mode 1 {entry_pdbs} {temp_out} >{erro1} 2>{erro2}', shell = True)
  except Exception as err:
    print("Foldseek reported an error:\n\n"+str(err))
    log.write("\nFoldseek reported an error:\n\n"+str(err)+"\n")
    sys.exit

  from_foldseek_out = temp_out
  to_foldseek_out = f'{args.o}/foldseek_dir/{temp_out}.txt'

  try:
    os.rename(from_foldseek_out, to_foldseek_out)
  except:
    print(f"There was an error in the Foldseek running. Please check out:\n{erro1}\n{erro2}")
    sys.exit()

  from_foldseekdb_out = f'{temp_out}.dbtype'
  to_foldseekdb_out = f'{args.o}/foldseek_dir/{temp_out}.dbtype'

  try:
    os.rename(from_foldseekdb_out, to_foldseekdb_out)
  except:
    print(f"There was an error in the Foldseek running. Please check out:\n{erro1}\n{erro2}")
    sys.exit()

  entry = f'{args.o}/foldseek_dir/{temp_out}.txt'
  file_AA_3Di = f"{args.o}/{args.s}_AA_3Di.fasta"
  file_3Di = f"{args.o}/{args.s}_3Di.fasta"

  log.write("""File 3Di characters: {file_3Di}
File amino acids + 3Di characters: {file_AA_3Di}""")

  with open(entry, "r") as foldseek_file:
    lines = foldseek_file.readlines()

  with open(file_AA_3Di, "w") as open_AA_3Di:
    with open(file_3Di, "w") as open_3Di:
      for line in lines:
        if ".pdb" in line.split('\t')[0]:
          seq_name = line.split('\t')[0].split('.pdb')[0]
        else:
          seq_name = line.split('\t')[0]
        seq_AA = line.split("\t")[1]
        seq_3Di = line.split("\t")[2]
        open_3Di.write(f">{seq_name}\n{seq_3Di}\n")
        open_AA_3Di.write(f">{seq_name}\n{seq_AA}{seq_3Di}\n")

def run_needle3Di(fasta_3Difile, output_dir):
  if fasta_3Difile.endswith("AA_3Di.fasta"):
    save_needle3Di_dir = f"{output_dir}/needle2.5Di_dir"
    log.write(f"\nSaving in: {save_needle3Di_dir}")
  else:
    save_needle3Di_dir = f"{output_dir}/needle3Di_dir"
    log.write(f"\nSaving in: {save_needle3Di_dir}")
  if os.path.isdir(save_needle3Di_dir) == False:
    os.mkdir(save_needle3Di_dir)

  seqs = []
  actual_seq = []
  with open(fasta_3Difile, 'r') as file_3Di:
    for line in file_3Di:
      line = line.strip()
      if line.startswith(">"):
        if actual_seq:
          seqs.append(actual_seq)
        actual_seq = [line]
      else:
        actual_seq.append(line)

    if actual_seq:
      seqs.append(actual_seq)

  if fasta_3Difile.endswith("AA_3Di.fasta"):
    needle_file = save_needle3Di_dir+"/"+output_dir+"_2.5Di_simil.needle"
  else:
    needle_file = save_needle3Di_dir+"/"+output_dir+"_3Di_simil.needle"
  stdout_needle_file = save_needle_dir+"/"+output_dir+"_stdout.txt"
  with open(stdout_needle_file, 'w') as open_needle:
    open_needle.write("")
  with open(needle_file, 'w') as open_needle:
    open_needle.write("")

  if fasta_3Difile.endswith("AA_3Di.fasta"):
    log.write("\nCommand model of modified Needle:\n")
    log.write(f"/home/geninfo/argruber/bin/scripts/3di_needle -kaa 1.4 -kdi 2.1 -debug -asequence sequence.fasta -bsequence restofsequences.fasta -gapopen 8.0 -gapextend 2.0 -outfile temp_out_file.needle -adatafile /home/geninfo/argruber/bin/EMBOSS-6.6.0/emboss/data/EBLOSUM62 -bdatafile /home/geninfo/argruber/bin/EMBOSS-6.6.0/emboss/data/mat3di.out\,")
  else:
    log.write("\nCommand model of Needle:\n")
    log.write(f"needle -asequence sequence.fasta -bsequence restofsequences.fasta -gapopen 8.0 -gapextend 2.0 -datafile 3Di_matrix.txt -aformat3 pair -outfile temp_out_file.needle\n")

  for seq in seqs:
    in_seq = seq[0][1:].split(' ')[0]
    in_seq_file = seq[0][1:].split(' ')[0]+'.fasta'
    out_seq_file = seq[0][1:].split(' ')[0]+'.needle'
    with open(in_seq_file, 'w') as file_3Di:
      file_3Di.write('\n'.join(seq))
    with open('comparated_sequences.fasta', 'w') as file_3Di:
      for l in seqs[seqs.index(seq):]:
        file_3Di.write('\n'.join(l)+'\n')
    try:
      if fasta_3Difile.endswith("AA_3Di.fasta"):
        subprocess.call(f"/home/geninfo/argruber/bin/scripts/3di_needle -kaa 1.4 -kdi 2.1 -debug -asequence {in_seq_file} -bsequence comparated_sequences.fasta -gapopen 8.0 -gapextend 2.0 -outfile {out_seq_file} -adatafile /home/geninfo/argruber/bin/EMBOSS-6.6.0/emboss/data/EBLOSUM62 -bdatafile /home/geninfo/argruber/bin/EMBOSS-6.6.0/emboss/data/mat3di.out >{stdout_needle_file}", shell = True)
      else:
        subprocess.call(f"needle -asequence {in_seq_file} -bsequence comparated_sequences.fasta -gapopen 8.0 -gapextend 2.0 -datafile 3Di_matrix.txt -aformat3 pair -outfile {out_seq_file} >{stdout_needle_file}", shell = True)
    except:
      print("Error at running Needle.")
      sys.exit
    else:
      print(f"Running Needle with {in_seq}...")
      time.sleep(1)
    os.remove(in_seq_file)
    os.remove('comparated_sequences.fasta')
    with open(out_seq_file, 'r') as open_moment_needle:
      with open(needle_file, 'a') as open_needle:
        for line in open_moment_needle:
          open_needle.write(line)
    os.remove(out_seq_file)

def data_from_needle3Di(file, type_sequence):
  data_score = []
  with open(file, 'r') as open_needle:
    pair_score = []
    for needle_line in open_needle:
      if needle_line.startswith("# 1:"):
        pair_score.append(needle_line[5:-1])
      elif needle_line.startswith("# 2:"):
        pair_score.append(needle_line[5:-1])
      if type_sequence == "3di":
        if needle_line.startswith("# Similarity:"):
          pair_score.append(round(float(needle_line.split("(")[1].split("%")[0]), 2))
      elif type_sequence == "2.5di":
        if needle_line.startswith("# Score:"):
          pair_score.append(float(needle_line.split(": ")[1]))
      else:
        continue
      if len(pair_score) == 3:
        data_score.append(pair_score)
        pair_score = []
  return data_score

def create_clustermap(all_data, args):
  for data in all_data:
    print("Generating clustermap file: "+data)

    if args.n == "id" and args.p == "tm":
      for o in all_data[data]:
        for t in o:
          if t != o[2]:
            if t[2] == "_":
              all_data[data][all_data[data].index(o)][o.index(t)] = "_".join(t.split("_")[0:2])
            else:
              all_data[data][all_data[data].index(o)][o.index(t)] = t.split("_")[0]


    if args.n == "name" or args.n == "all" and os.path.isdir(args.i) == False:
      new_data = []
      with open(args.i, "r") as read_fasta:
        for line in read_fasta:
          if line.startswith(">"):
            new_data.append([line[0][1:].split(' ')[0], line.split("[")[1].rsplit("]")[0]])
      for o in all_data[data]:
        for t in o:
          for n in new_data:
            if t == n[0]:
              if args.n == "name":
                all_data[data][all_data[data].index(o)][o.index(t)] = n[1]
              elif args.n == "all":
                all_data[data][all_data[data].index(o)][o.index(t)] = n[0]+" - "+n[1]

#    print(all_data)

    order = []
    for l in all_data[data]:
      if l[0] not in order:
        order.append(l[0])

    # Gera nome do arquivo final.
    jpg_file_name = data + '.jpg'
    svg_file_name = data + '.svg'

    # Cria matriz vazia.
    matriz = []
    for k in range(len(order)):
        linha = [None] * len(order)
        matriz.append(linha)

    # Posiciona os valores de similaridade na matriz de acordo com seus respectivos códigos, guiados pela ordem dos códigos do arquivo de referência.
    for element in all_data[data]:
        i = order.index(element[0])
        j = order.index(element[1])
        if i < j:
            h = i
            i = j
            j = h
        matriz[j][i] = matriz[i][j] = element[2]

    table = {}
    table[''] = order
    for l in range(0, len(matriz)):
      table[order[l]] = matriz[l]

    final_table = pd.DataFrame(table)
    final_table.set_index('', inplace=True)

    color = args.c
    if sns.color_palette(color):
      color_msa = color
      if color.endswith("_r"):
        color_pa = color[:-2]
      else:
        color_pa = color+"_r"
    else:
      print("Error: Invalid color (-c).")
      sys.exit()

    if args.l == "yes":
      linew = 0.15
    elif args.l == "no":
      linew = 0

    if jpg_file_name.endswith("_mldist.jpg"):
      clustermap = sns.clustermap(final_table, method='complete', cmap=color_msa,
                   xticklabels=True, yticklabels = True, linewidths=linew,
                   vmin=0, vmax=9, cbar_pos=(.02, .32, .03, .2),
                   figsize=(25, 25))

    elif jpg_file_name.endswith("_simil.jpg") or jpg_file_name.endswith("_ident.jpg") or jpg_file_name.endswith("_3Di.jpg"):
      clustermap = sns.clustermap(final_table, cmap=color_pa,
                   xticklabels=True, yticklabels = True,
                   cbar_pos=(.02, .32, .03, .2), linewidths=linew,
                   vmin=0, vmax=100,
                   figsize=(25, 25))

    elif jpg_file_name.endswith("_tmscores.jpg"):
      clustermap = sns.clustermap(final_table, cmap=color_pa,
                   xticklabels=True, yticklabels = True,
                   cbar_pos=(.02, .32, .03, .2), linewidths=linew,
                   vmin=0, vmax=1,
                   figsize=(25, 25))


    elif jpg_file_name.endswith("_2.5Di.jpg"):
      max_score = 0
      for n in all_data[data]:
        if n[0] != n[1]:
          if n[2] > max_score:
            max_score = n[2]
      clustermap = sns.clustermap(final_table, cmap=color_pa,
                   xticklabels=True, yticklabels = True,
                   cbar_pos=(.02, .32, .03, .2), linewidths=linew,
                   vmin=0, vmax=max_score,
                   figsize=(25, 25))


    clustermap.ax_row_dendrogram.set_visible(False)

    plt.savefig(jpg_file_name, dpi=200)
    plt.savefig(svg_file_name, dpi=200, format='svg')

    plt.clf()

def create_freqplot(all_data):
  for data in all_data:
    print(f"Generating frequency plot file: {data}_freqplot")

    jpg_file_name = data + '_freqplot.jpg'
    svg_file_name = data + '_freqplot.svg'

    if data.endswith("_ident") or data.endswith("_simil") or data.endswith("_3Di"):
      freqplot_data = {numero: 0 for numero in range(101)}
      for n in all_data[data]:
        if n[0] != n[1]:
          if round(n[2]) in freqplot_data:
            freqplot_data[round(n[2])] += 1

    elif data.endswith("_tmscores"):
      freqplot_data = {numero/100: 0 for numero in range(101)}
      for n in all_data[data]:
        if n[0] != n[1]:
          if round(n[2], 2) in freqplot_data:
            freqplot_data[round(n[2], 2)] += 1

    elif data.endswith("_mldist"):
      freqplot_data = {numero: 0 for numero in [round(x * 0.1, 1) for x in range(91)]}
      for n in all_data[data]:
        if n[0] != n[1]:
          if round(n[2], 1) in freqplot_data:
            freqplot_data[round(n[2], 1)] += 0.5
    elif data.endswith("_2.5Di"):
      max_score = 0
      for n in all_data[data]:
        if n[0] != n[1]:
          if round(n[2]) > max_score:
            max_score = round(n[2])
      freqplot_data = {numero: 0 for numero in range(101)}
      for n in all_data[data]:
        if n[0] != n[1]:
          if round(n[2]/max_score, 2)*100 in freqplot_data:
            freqplot_data[round(n[2]/max_score, 2)*100] += 1

    freqplot_data = pd.DataFrame.from_dict(freqplot_data, orient = 'index')
    freqplot_data = freqplot_data.rename(columns={0: ""})

    plt.subplots(figsize=(6.4, 4.8))

    freqplot = sns.lineplot(data=freqplot_data, palette='Greens_d')

    if data.endswith("_ident"):
      freqplot.set_xlabel('Identity (%)')
      freqplot.set_title('Identity frequency plot')
    elif data.endswith("_simil"):
      freqplot.set_xlabel('Similarity (%)')
      freqplot.set_title('Similarity frequency plot')
    elif data.endswith("_mldist"):
      freqplot.set_xlabel('Maximum-likelihood distance')
      freqplot.set_title('Maximum-likelihood distance frequency plot')
    elif data.endswith('_tmscores'):
      freqplot.set_xlabel('TM-scores')
      freqplot.set_title('TM-scores frequency plot')
    elif data.endswith('_3Di'):
      freqplot.set_xlabel('3Di similarity')
      freqplot.set_title('3Di similarity frequency plot')
    elif data.endswith('_2.5Di'):
      freqplot.set_xlabel(f'2.5Di score ({str(max_score)})')
      freqplot.set_title('2.5Di scores frequency plot')
    freqplot.set_ylabel('Frequency in numbers')

    def num2freq(x):
      return 100*(x/(len(all_data[data])/2))
    def freq2num(x):
      return (x/100)*(len(all_data[data])/2)
    secax = freqplot.secondary_yaxis('right', functions=(num2freq, freq2num))
    secax.set_ylabel('Frequency (%)')

    plt.savefig(jpg_file_name, dpi=200)
    plt.savefig(svg_file_name, dpi=200, format='svg')

    plt.clf()

def not_recognized_methods(args):
    nrecognmethods = ""
    for m in args.p:
      if m not in "als32":
        nrecognmethods += m
    if len(nrecognmethods) != 0:
      print(f"Methods '{nrecognmethods}' not recognized.")

if __name__ == '__main__':
  if not len(sys.argv)>1:
    print(help)
  elif args.help == True:
    print(help)
  elif args.version == True:
    print("""
3A-DGT - Triple-A DGT version """+version+""" - 08 nov 2023
The All-Against-All Distance Graphic Tool
(c) 2023. Igor Custódio dos Santos & Arthur Gruber
""")
  else:
#    mandatory_param_check(args)
    check_param(args)
    cmd_iqtree = check_iqtree(args.q)
    cmd_mafft = check_mafft(args.f)
    not_recognized_methods(args)
    type_header, type_sequence = check_fasta(args.i)
    check_pdb(args.s)
  #  else:
   #   type_header = 'not_ncbi'
    #  type_sequence = "PDB"
    args.o = check_output_dir(args.o)
    save_graphics_dir, logfile = where_to_save_graphics(args.o)
    save_align_dir = f"{args.o}/mafft_dir"
    save_needle_dir = f"{args.o}/needle_dir"
    save_iqtree_dir = f"{args.o}/iqtree_dir"
    save_tmalign_dir = f"{args.o}/TMalign_dir"
    save_3Dineedle_dir = f"{args.o}/needle3Di_dir"
    save_25dineedle_dir = f"{args.o}/needle2.5Di_dir"

#    with open(f'{args.o}/stdout.log', 'w') as sys.stdout:

    if args.i != None and os.path.isdir(args.i) == False:
      args.i = entry_file(args, type_header)
    with open(logfile, "w") as log:
      log_header(log, args, type_sequence, type_header, version)

      os.mkdir(save_graphics_dir)
      all_data = {}

      part1 = "no"
      if "a" in args.p or "l" in args.p:
        part1 = "yes"
      if args.i == None and part1 == "yes":
        print("""
WARNING:
Not possible to run pairwise aa sequence alignment ('a') 
or maximum likelihood ('l').
FASTA file (-i) not specified.
""")
      else:

        if "a" in args.p:
          log.write("""
Pairwise Alignment:
""")
          print("\nMethod: pairwise sequence alignment with NW")
          if os.path.isfile(save_needle_dir+"/"+args.o+".needle"):
            log.write("""
- Needle file already exists. Skipping Needle run.
- Using data from: """+save_needle_dir+"/"+args.o+".needle\n")
            print("Needle file already exists. Skipping Needle run.")
          else:
            log.write("\nNeedle file does not exist yet. Running Needle:\n")
            print("Running pairwise sequence alignments with needle…")
            run_needle(args.i, args.o, type_header, type_sequence, log)
            print("Pairwise sequence alignments completed.")
          data = data_from_needle(save_needle_dir+"/"+args.o+".needle", type_sequence)
          for subdata in data:
            if data.index(subdata) == 0:
              matrix_name = save_needle_dir+"/"+args.o+"_ident_matrix.csv"
              save_matrix(subdata, matrix_name, log)
            if data.index(subdata) == 1:
              matrix_name = save_needle_dir+"/"+args.o+"_simil_matrix.csv"
              save_matrix(subdata, matrix_name, log)
          if len(data) == 1:
            data[0] = correct_label(data[0], args)
            all_data[save_graphics_dir+"/"+args.o+"_ident"] = data[0]
          else:
            data[0] = correct_label(data[0], args)
            all_data[save_graphics_dir+"/"+args.o+"_ident"] = data[0]
            data[1] = correct_label(data[1], args)
            all_data[save_graphics_dir+"/"+args.o+"_simil"] = data[1]
          print("Generating heatmap plot of pairwise sequence alignment distance…")
          create_clustermap(all_data, args)
          print("Done.")
          print("Generating frequency distribution plot of pairwise sequence alignment distance…")
          create_freqplot(all_data)
          print("Done.")
          all_data = {}

        if "l" in args.p:
          log.write("""
Maximum Likelihood:
""")
          print("\nMethod: maximum likelihood")
          if os.path.isfile(save_align_dir+"/"+args.o+".align"):
            log.write("""
- Multiple sequences alignment file already exists. Skipping MAFFT run.
- Using data from: {save_align_dir}/{args.o}.align\n""")
            print("MAFFT aligned file already exists. Skipping MAFFT step.")
          else:
            log.write("\nMultiple sequences alignment file does not exist yet. Running MAFFT:\n")
            print("Running multiple sequence alignment with MAFFT…")
            run_mafft(args.i, args.o, cmd_mafft, log)
            print("Multiple sequence alignment completed.")
          if os.path.isfile(save_iqtree_dir+"/"+args.o+".mldist"):
            log.write("""
- IQ-TREE maximum-likelihood distance matrix file already exists. Skipping IQ-TREE run.
- Using data from: {save_iqtree_dir}/{args.o}.mldist\n""")
            print("IQ-TREE maximum-likelihood distance matrix file already exists. Skipping IQ-TREE step.")
          else:
            log.write("\nIQ-TREE maximum-likelihood distance matrix file does not exist yet. Running IQ-TREE:\n")
            print("Running phylogenetic analysis IQ-TREE…")
            run_iqtree(save_align_dir+"/"+args.o+".align", args.o, cmd_iqtree, log)
            print("Phylogenetic analysis completed.")

          data = data_from_mldist(save_iqtree_dir+"/"+args.o+".mldist", log)
          data = correct_label(data, args)
          all_data[save_graphics_dir+"/"+args.o+"_mldist"] = data
          print("Generating heatmap plot of maximum likelihood distance…")
          create_clustermap(all_data, args)
          print("Done.")
          print("Generating frequency distribution plot of pairwise alignment distance…")
          create_freqplot(all_data)
          print("Done.")
          all_data = {}


      part2 = "no"
      if "s" in args.p or "3" in args.p or "2" in args.p:
        part2 = "yes"
      if args.s == None and part2 == "yes":
        print("""
WARNING:
Not possible to run pairwise structural alignment ('s'),
pairwise 3di character alignment ('3') or combined
alignment of 3di and amino acid sequences ('2').
PDB files directory (-p) not specified.
""")
      else:
        if args.s != None and args.s.endswith("/"):
          args.s = args.s[:-1]

        if "s" in args.p:
          print("\nMethod: pairwise structural alignment")
          log.write("""
Structural alignmente (TM-align):
""")
          print("Running pairwise structural alignments with TM-align")
          run_tmalign(args.s, args.o, log)
          print("Pairwise structural alignments completed.")

          data = data_from_tmalign(args.o)
          matrix_name = save_tmalign_dir+"/"+args.o+"_tmscores_matrix.csv"
          save_matrix(data, matrix_name, log)
          all_data[save_graphics_dir+"/"+args.o+"_tmscores"] = data

          print("Generating heatmap plot of pairwise structural alignment distance…")
          create_clustermap(all_data, args)
          print("Done.")

          print("Generating frequency distribution plot of pairwise structural alignment distance…")
          create_freqplot(all_data)
          print("Done.")
          all_data = {}


        if "3" in args.p or "2" in args.p:
          log.write("\nFoldseek step run:\n")
          file_AA_3Di = f"{args.o}/{args.s}_AA_3Di.fasta"
          file_3Di = f"{args.o}/{args.s}_3Di.fasta"
          if os.path.isfile(file_AA_3Di) == False and os.path.isfile(file_3Di) == False:
            print("Converting PDB to 3Di-character sequences with Foldseek")
            run_foldseek(args, log)
            print("Conversion completed.\n")
          else:
            log.write("""\nSkipping Foldseek run. Output files already exists:
- {file_3Di}
- {file_AA_3Di}\n""")
            print("3Di and aa/3Di FASTA files already exist. Skipping Foldseek run.")


          if "3" in args.p:
            print("\nMethod: pairwise 3Di-character alignment")
            log.write("""
3Di Similarity:
""")
            needle_file = f'{args.o}/needle3Di_dir/{args.o}_3Di_simil.needle'
            if os.path.isfile(needle_file) == False:
              print("Running pairwise 3Di-character sequence alignments with needle…")
              run_needle3Di(file_3Di, args.o)
              print("Pairwise 3Di-character sequence alignments completed.")
            else:
              log.write("\nNeedle 3Di similarity file already exists. Skipping Needle run.\n")
              print("Needle 3Di similarity file already exists. Skipping Needle run.")
            data = data_from_needle3Di(needle_file, "3di")
            matrix_name = save_3Dineedle_dir+"/"+args.o+"_3Di_simil_matrix.csv" 
            save_matrix(data, matrix_name, log) 
            all_data[save_graphics_dir+"/"+args.o+"_3Di"] = data

            print("Generating heatmap plot of 3Di-character pairwise sequence alignment distance…")
            create_clustermap(all_data, args)
            print("Done.")

            print("Generating frequency distribution plot of 3Di-character pairwise structural alignment distance…")
            create_freqplot(all_data)
            print("Done.")
            all_data = {}


          if "2" in args.p:
            print("\nMethod: pairwise combined aa/3Di-character alignment")
            log.write("""
3Di + amino acids (2.5Di) Combined Similarity:
""")
            needle_file = f'{args.o}/needle2.5Di_dir/{args.o}_2.5Di_simil.needle'
            if os.path.isfile(needle_file) == False:
              print("Running pairwise combined aa/3Di-character sequence alignments with 2.5Di_needle…")
              run_needle3Di(file_AA_3Di, args.o)
              print("Pairwise combined aa/3Di-character sequence alignments completed.")
            else:
              log.write("\nNeedle aa/3Di similarity file already exists. Skipping modified Needle run.\n")
              print("Needle aa/3Di similarity file already exists. Skipping modified Needle run.") 
            data = data_from_needle3Di(needle_file, "2.5di")
            matrix_name = save_25dineedle_dir+"/"+args.o+"_2.5Di_simil_matrix.csv"
            save_matrix(data, matrix_name, log)
            all_data[save_graphics_dir+"/"+args.o+"_2.5Di"] = data

            print("Generating heatmap plot of pairwise combined aa/3Di-character sequence alignment distance…")
            create_clustermap(all_data, args)
            print("Done.")

            print("Generating frequency distribution plot of 3Di-character pairwise structural alignment distance…")
            create_freqplot(all_data)
            print("Done.")
            all_data = {}

#      print(all_data)

#        os.mkdir(save_graphics_dir)
#        print("all_data: ")
#        print(all_data)
     #   create_freqplot(all_data)
      #  create_clustermap(all_data, args)



