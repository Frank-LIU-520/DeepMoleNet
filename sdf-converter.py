# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:47:15 2020

@author: lzt
"""
import pathlib

from openbabel.pybel import (readfile,Outputfile) 

'''
def MolFormatConversion(input_file:str,output_file:str,input_format="xyz",output_format="sdf"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
    output_file_writer.close()
    print('%d molecules converted'%(i+1))
'''    
file_name=[]
sdf_dir = pathlib.Path('./')
for sdf_file in sdf_dir.glob("*.xyz"):
    file_name.append(sdf_file.name)
   # print(file_name)
    
print(len(file_name))    
    
def MolFormatConversion(input_file:str,output_file:str,input_format="xyz",output_format="sdf"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
    output_file_writer.close()
    print('%d molecules converted'%(i+1))



for i in file_name:
    input_file=i.split('.')[0]+".xyz"
    output_file=i.split('.')[0]+".sdf"
    MolFormatConversion(input_file, output_file)

#MolFormatConversion("2.xyz", "2.sdf")
