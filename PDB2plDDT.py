#! /usr/bin/python3

import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from argparse import RawTextHelpFormatter
import sys

version = "1.00"

help = """
PDB2plDDT version """+version+""" - 26 Mar 2024
PDB to plDDT plot
(c) 2023. Igor Custódio dos Santos & Arthur Gruber

Usage: PDB2plDDT.py -i <PDB directory> -o <output>

-i pdb <directory name>         Structure file directory (PDB directory)
-o output <directory name>      Output directory to store files
				(default: plDDT_1)
"""

parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
parser.add_argument('-i')
parser.add_argument('-o')
parser.add_argument('-h', '--help', action='store_true')
parser.add_argument('-v', '--version', action='store_true')
args = parser.parse_args()

def no_empty(x):
  while '' in x:
    x.remove('')
  return x

def mandatory_param_check(args):
  not_specified = []
  if args.i == None:
    not_specified.append("-i [input_file]")

  if len(not_specified) != 0:
    if len(not_specified) == 1:
      not_specified = not_specified[0]
    else:
      not_specified = ", ".join(not_specified)
    print("""Error: The command is missing mandatory parameters: """+not_specified+""".
Use the command template below adding any optional parameters you want.

PDB2plDDT.py -i [PDB files directory] -o [output]""")
    sys.exit()

def where_to_save_graphics(output):
  if output == None:
    i = 1
    while True:
      if os.path.isdir("plDDT_"+str(i)):
        i +=1
        continue
      else:
        save_dir = "plDDT_"+str(i)
        break
    print(f"Output directory not specified. Saving files in {save_dir}")
    return save_dir
  else:
    save_dir = output
    return save_dir

def take_plDDT(input_dir, output):
  if os.path.isdir(output) == False:
    os.mkdir(output)

  file_list = os.listdir(input_dir)

  for pdb in file_list:
    if pdb.endswith(".pdb")==False:
      print(f'In "{input_dir}" directory, the file "{pdb}" might not be a pdb format file.\nPlease take it off from the input directory or rename it with pdb extention.')
      sys.exit()

  for file in file_list:
    save_file_jpg = f"{output}/{file[:-4]}.jpg"
    save_file_svg = f"{output}/{file[:-4]}.svg"
    save_file_txt = f"{output}/{file[:-4]}.txt"

    with open(f"{args.i}/{file}", "r") as open_file:
      lines = open_file.readlines()

    data = {}
    for line in lines:
      if line.startswith("ATOM"):
        if no_empty(line.split(" "))[2] == "CA":
          data[int(no_empty(line.split(" "))[5])] = float(no_empty(line.split(" "))[10])

    freqplot_data = pd.DataFrame.from_dict(data, orient = 'index')
    freqplot_data = freqplot_data.rename(columns={0: ""})

    freqplot = sns.lineplot(data=freqplot_data)
    freqplot.set_xlabel('Position of amino acid residues')
    if " " in file:
      freqplot.set_title(f'{file.split(" ")[0]} - plDDT plot')
    else:
      freqplot.set_title(f'{file[:-4]} - plDDT plot')

    def num2conf(x):
      return x
    def conf2num(x):
      return x
    secax = freqplot.secondary_yaxis('left', functions=(num2conf, conf2num))
    secax.set_ylabel('plDDT (%)')

    plt.ylim(0, 100)
#    plt.savefig(save_file_jpg, dpi = 200)
#    plt.savefig(save_file_svg, format='svg')

    plt.clf()

    dataVeryHigh = 0
    dataHigh = 0
    dataLow = 0
    dataVeryLow = 0
    sum = 0
    for d in data:
      sum += data[d]
      if data[d]>=90:
        dataVeryHigh += 1
      elif data[d]>=70 and data[d]<90:
        dataHigh += 1
      elif data[d]>=50 and data[d]<70:
        dataLow += 1
      elif data[d]<50:
        dataVeryLow += 1

    with open(save_file_txt, "w") as write_out:
      write_out.write(f"""
plDDT values (PDB2plDDT)

  - PDB file: {file}
  - Number of residues: {str(len(data))}
  - Average plDDT value: {str(round(sum/len(data), 2))}

plDDT ranges:

  * Very high (plDDT > 90):	{dataVeryHigh} ({str(round(dataVeryHigh/len(data), 3))})
  * High (90 > plDDT > 70):	{dataHigh} ({str(round(dataHigh/len(data), 3))})
  * Low (70 > plDDT > 50):	{dataLow} ({str(round(dataLow/len(data), 3))})
  * Very low (plDDT < 50):	{dataVeryLow} ({str(round(dataVeryLow/len(data), 3))})

""")

def comp_all(input_dir, output):
  file_list = os.listdir(output)

  d = 0
  list_average = []
  list_residues = []

  list_veryhigh = []
  list_veryhigh_n = []

  list_high = []
  list_high_n = []

  list_low = []
  list_low_n = []

  list_verylow = []
  list_verylow_n = []

  for file in file_list:
    if file.endswith(".txt"):
      d += 1
      with open(f"{output}/{file}", "r") as open_file:
        lines = open_file.readlines()
      for line in lines:

        if "* Very high (plDDT > 90):" in line:
          list_veryhigh.append(float(line.split("(")[2].split(")")[0]))
          list_veryhigh_n.append(int(line.split("\t")[1].split(" (")[0]))

        elif "* High (90 > plDDT > 70):" in line:
          list_high.append(float(line.split("(")[2].split(")")[0]))
          list_high_n.append(int(line.split("\t")[1].split(" (")[0]))

        elif "* Low (70 > plDDT > 50):" in line:
          list_low.append(float(line.split("(")[2].split(")")[0]))
          list_low_n.append(int(line.split("\t")[1].split(" (")[0]))

        elif "* Very low (plDDT < 50):" in line:
          list_verylow.append(float(line.split("(")[2].split(")")[0]))
          list_verylow_n.append(int(line.split("\t")[1].split(" (")[0]))

        elif "- Average plDDT value:" in line:
          list_average.append(float(line.split(": ")[1].split("\n")[0]))

        elif "- Number of residues:" in line:
          list_residues.append(float(line.split(": ")[1].split("\n")[0]))


    with open(f"{output}/all.txt", "w") as write_out:
      write_out.write(f"""
plDDT values (PDB2plDDT) - Compilation file

  - PDB directory: {input_dir}
  - Average of number of residues: {str(round(sum(list_residues)/d, 3))}
  - Average of average plDDT value: {str(round(sum(list_average)/d, 3))}

plDDT ranges (Average):

  * Very high (plDDT > 90):	{str(round(sum(list_veryhigh_n)/d, 3))} ({str(round(sum(list_veryhigh)/d, 3))})
  * High (90 > plDDT > 70):	{str(round(sum(list_high_n)/d, 3))} ({str(round(sum(list_high)/d, 3))})
  * Low (70 > plDDT > 50):	{str(round(sum(list_low_n)/d, 3))} ({str(round(sum(list_low)/d, 3))})
  * Very low (plDDT < 50):	{str(round(sum(list_verylow_n)/d, 3))} ({str(round(sum(list_verylow)/d, 3))})

""")

if __name__ == '__main__':
  if not len(sys.argv)>1:
    print(help)
  elif args.help == True:
    print(help)
  elif args.version == True:
    print("""
PDB2plDDT version """+version+""" - 25 mar 2024
(c) 2024. Igor Custódio dos Santos & Arthur Gruber
""")
  else:
    if args.i.endswith("/"):
      args.i = args.i.replace("/", "")
    mandatory_param_check(args)

    args.o = where_to_save_graphics(args.o)

    if os.path.isdir(args.i) == False:
      print(f"ERROR: {args.i} is not a directory.")
      sys.exit()
    else:
      take_plDDT(args.i, args.o)
      comp_all(args.i, args.o)

