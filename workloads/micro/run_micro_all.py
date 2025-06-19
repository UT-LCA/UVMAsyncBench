import os
import argparse
import sys
import subprocess
import psutil

import os
import collections
import csv

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import xlsxwriter
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerTuple

from subprocess import Popen, PIPE

from scipy.stats import gmean

patterns = ["", "", "", "", "", ""]
color_tab = ['#000000', '#0000ff', '#ff0000', '#ff6666', '#00ff00', '#00ffcc']
edge_color_tab = ['#000000', '#000000', '#000000', '#000000', '#000000', '#000000']

prefix = 'run_'
parameter_super_list = ['large', 'super', 'mega']
config_super_list = ['standard', 'async', 'uvm_prefetch', 'uvm_prefetch_async', 'pinned', 'pinned_async']
workload_super_list = ['vector_seq', 'vector_rand', 'saxpy', 'gemv', 'gemm', '2DCONV', '3DCONV']


def dict_to_list(input_dict):
    return_list = []
    for elemement in input_dict:
        return_list.append(elemement)
    return return_list

def addOptions(parser):
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="Number of iterations")
    parser.add_argument("-g", "--gpu", type=str, default='a100',
                        help="GPU Platform")
    parser.add_argument("-o", "--output", type=str, default='output.xlsx',
                        help="output trace log file")
    parser.add_argument("-f", "--figure", type=str, default='micro',
                        help="output pdf file")
    parser.add_argument("-p", "--profiling", action='store_true',
                        help="whether to run profiling or just parse results")
    parser.add_argument("-c", "--clean", action='store_true',
                        help="whether to clean all results")


def get_config_list(root_directory):
    config_list = []
    for dict in os.listdir(root_directory):
        if os.path.isdir(dict) and dict in config_super_list:
            config_list.append(dict)
    return config_list


def get_workload_dict(root_directory, config_list):
    workload_list = []
    workload_dict = dict()
    for config in config_list:
        config_dir = root_directory + '/' + config
        
        for root, directories, files in os.walk(config_dir, topdown=False):
            for dir in directories:
                if dir in workload_super_list:
                    if dir not in workload_dict:
                        workload_dict[dir] = dict()
                    workload_dict[dir][config] = os.path.join(root, dir)    
                    if dir not in workload_list:
                        workload_list.append(dir)
                
    return workload_list, workload_dict

def get_trace_workload_dict(root_directory, config_list):
    workload_list = []
    workload_dict = dict()
    
    for dir in os.listdir(root_directory):
        if dir in workload_super_list:
            if dir not in workload_dict:
                workload_dict[dir] = dict()
            workload_dir = os.path.join(root_directory, dir)
            for config in config_list:
                workload_dict[dir][config] = os.path.join(workload_dir, config)    
            if dir not in workload_list:
                workload_list.append(dir)
    return workload_list, workload_dict

def execute_bashes(workload_dict, gpu, iterations):
    os.system('mkdir -p ../../traces/')
    os.system('mkdir -p ../../traces/micro')
    for workload in workload_dict:
        if workload in workload_super_list:
            for config in workload_dict[workload]:
                os.system('mkdir -p ../../traces/micro/' + config)
                os.system('mkdir -p ../../traces/micro/' + config + '/' + workload)
                if config in config_super_list:
                    cur_dir = workload_dict[workload][config]
                    pwd = os.getcwd()
                    os.chdir(cur_dir)
                    os.system('make clean')
                    os.system('make')
                    for para in parameter_super_list:
                        for i in range(0, iterations):
                            sh_file = './' + prefix + para + '.sh'
                            exe_cmd = sh_file + ' > ' + para + '_' + gpu + '_' + str(i) + '.log'
                            # if gpu == 'a100':
                            #     exe_cmd = sh_file + ' > ' + para + '_' + gpu + '_' + str(i) + '.log'
                            # else:
                            #     exe_cmd = 'numactl -m 1 -N 1 ' + sh_file + ' > ' + para + '_' + gpu + '_' + str(i) + '.log'
                            os.system(exe_cmd)
                    os.chdir(pwd)
                    os.system('mv ' + cur_dir + '/*.log ../../traces/micro/' + config + '/' + workload + '/' )


def execute_clean_bashes(workload_dict):
    for workload in workload_dict:
        if workload in workload_super_list:
            for config in workload_dict[workload]:
                if config in config_super_list:
                    cur_dir = workload_dict[workload][config]
                    pwd = os.getcwd()
                    os.chdir(cur_dir)
                    os.system('make clean')
                    os.system('rm *.log')
                    os.chdir(pwd)

def process_file(log_file, config):
    result_dict = dict()
    text = open(log_file, "r")

    overlap = 0

    result_dict['gpu_kernel'] = 0
    result_dict['memcpy'] = 0
    result_dict['memcpy_HtoD'] = 0
    result_dict['memcpy_DtoH'] = 0
    result_dict['allocation'] = 0

    result_dict['malloc'] = 0
    result_dict['free'] = 0

    for line in text:
        line = line.replace(':', '')
        line = line.strip()
        words = line.split(',')
        
        if 'KERNEL' in words[0] and len(words) >= 4:
            result_dict['gpu_kernel'] += int(words[-1])
        elif 'MEMCPY' in words[0]:
            if 'HTOD' in words[0] or 'HtoD' in words[0]:
                result_dict['memcpy_HtoD'] += int(words[-1])
            else:
                result_dict['memcpy_DtoH'] += int(words[-1])
        elif 'cudaMalloc' in words[0]:
            result_dict['allocation'] += int(words[3])
            result_dict['malloc'] += int(words[3])
        elif 'cudaFree' in words[0]:
            result_dict['allocation'] += int(words[3])
            result_dict['free'] += int(words[3])
        
    return_dict = dict()
    
    if config == 'uvm':
        return_dict['gpu_kernel'] = result_dict['gpu_kernel'] - result_dict['memcpy_HtoD']
    else: 
        return_dict['gpu_kernel'] = result_dict['gpu_kernel']
    return_dict['memcpy'] = result_dict['memcpy_HtoD'] + result_dict['memcpy_DtoH']
    return_dict['allocation'] = result_dict['allocation']
    return_dict['malloc'] = result_dict['malloc']
    return_dict['free'] = result_dict['free']
        
    return return_dict


def process_results(workload_dict, gpu, iterations):
    result_dict = dict()
    for workload in workload_dict:
        if workload in workload_super_list:
            for config in workload_dict[workload]:
                if config in config_super_list:
                    cur_dir = workload_dict[workload][config]
                    for para in parameter_super_list:
                        if para not in result_dict:
                            result_dict[para] = dict()
                        if workload not in result_dict[para]:
                            result_dict[para][workload] = dict()
                            
                        # if config not in result_dict[para][workload]:
                        result_dict[para][workload][config] = []
                        for i in range(0, iterations):
                            log_file =  cur_dir + '/' + para + '_' + gpu + '_' + str(i) + '.log'
                            print(log_file)
                            result_dict[para][workload][config].append(process_file(log_file, config))
                        sorted(result_dict[para][workload])
                        sorted(result_dict[para])
    return result_dict
                    

def export_xlsx(result_dict, config_list, iterations, output_file):
    workbook = xlsxwriter.Workbook(output_file)
    
    for para in result_dict:        
        worksheet = workbook.add_worksheet(para)
        first_col = 'B'
        first_row = 3
            
        col_index = 0
        row_index = first_row
        for workload in result_dict[para]:
            if col_index + (ord(first_col) - ord('A')) < 26:
                current_col = chr(ord(first_col) + col_index)
            else:
                all = (ord(first_col) - ord('A')) + col_index
                dig_1 = all // 26 - 1
                dig_2 = all % 26
                current_col = chr(ord('A') + dig_1) + chr(ord('A') + dig_2)
            worksheet.write(current_col + '1', workload)
            for config in config_list:
                if col_index + (ord(first_col) - ord('A')) < 26:
                    current_col = chr(ord(first_col) + col_index)
                else:
                    all = (ord(first_col) - ord('A')) + col_index
                    dig_1 = all // 26 - 1
                    dig_2 = all % 26
                    current_col = chr(ord('A') + dig_1) + chr(ord('A') + dig_2)
                worksheet.write(current_col + '2', config)
                row_index = first_row
                result_list = dict_to_list(result_dict[para][workload][config][0])
                
                for i in range(0, iterations):
                    tmp_result_dict = dict()
                    for result in result_dict[para][workload][config][i]:
                        tmp_result_dict[result] = result_dict[para][workload][config][i][result]
                
                    if col_index == 0:
                        for j in range(0, len(result_list)):
                            worksheet.write('A' + str(j + row_index), result_list[j])
                
                    for j in range(0, len(result_list)):
                        worksheet.write(current_col + str(j + row_index), tmp_result_dict[result_list[j]])
                
                    row_index += len(result_list)
                col_index += 1
    workbook.close()


def plot_results(result_dict, config_list, workload_list, iterations, output_file):
    
    config_ordered_list = []
    for config in config_super_list:
        if config in config_list:
            config_ordered_list.append(config)
            
    workload_ordered_list = []
    for workload in workload_super_list:
        if workload in workload_list:
            workload_ordered_list.append(workload)
    
    for para in result_dict:        
        pandas_list = []
        pandas_list.append('workload')
        for config in config_list:
            pandas_list.append(config)

        pandas_dict = dict()
        for ele in pandas_list:
            pandas_dict[ele] = []

              
        for workload in result_dict[para]:
            for i in range(0, iterations):
                pandas_dict['workload'].append(workload)
                for config in config_list:
                    overall_time = 0
                    overall_time += result_dict[para][workload][config][i]['gpu_kernel']
                    overall_time += result_dict[para][workload][config][i]['memcpy']
                    overall_time += result_dict[para][workload][config][i]['allocation']
                    pandas_dict[config].append(overall_time)

        df = pd.DataFrame(pandas_dict)
        dd=pd.melt(df,id_vars='workload',value_vars=config_list,var_name='configs')

        my_pal = {"standard": '#000000', "async": '#0000ff', "uvm_prefetch": '#ff0000', "uvm_prefetch_async": '#ff6666', "pinned": '#00ff00', "pinned_async": '#00ff66'}
        # sns.set_theme(rc={'figure.figsize':(5.8,2.8)})
        matplotlib.rcParams['figure.figsize'] = 6.8,4.3
        sns.boxplot(data=dd, x='workload', y='value', hue='configs', order=workload_ordered_list, hue_order=config_ordered_list, palette=my_pal)
        # sns.tight_layout()
        plt.legend([], [], frameon=False)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
        # plt.legend(fontsize=14)

        plt.xticks(fontsize=14, rotation=15)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.xlabel("")
        plt.ylabel("Execution time (ns)", fontsize=14)
        plt.tight_layout()
        # plt.margins(x=0.01, y=0.01)
        
        plt.savefig(output_file + '_' + para + '_no_legend.pdf', bbox_inches='tight')
        plt.savefig(output_file + '_' + para + '_no_legend.png', bbox_inches='tight')
        plt.close()

def export_xlsx_all(result_dict, gpu, config_list, iterations, output_file):
    std_dict = dict()
    
    workbook = xlsxwriter.Workbook(output_file.replace(".xlsx", '') + '_all.xlsx')
    
    for para in result_dict:        
        worksheet = workbook.add_worksheet(para)
        first_col = 'B'
        first_row = 3
            
        col_index = 0
        row_index = first_row
        for workload in result_dict[para]:
            if col_index + (ord(first_col) - ord('A')) < 26:
                current_col = chr(ord(first_col) + col_index)
            else:
                all = (ord(first_col) - ord('A')) + col_index
                dig_1 = all // 26 - 1
                dig_2 = all % 26
                current_col = chr(ord('A') + dig_1) + chr(ord('A') + dig_2)
            worksheet.write(current_col + '1', workload)
            for config in config_list:
                if col_index + (ord(first_col) - ord('A')) < 26:
                    current_col = chr(ord(first_col) + col_index)
                else:
                    all = (ord(first_col) - ord('A')) + col_index
                    dig_1 = all // 26 - 1
                    dig_2 = all % 26
                    current_col = chr(ord('A') + dig_1) + chr(ord('A') + dig_2)
                worksheet.write(current_col + '2', config)
                row_index = first_row
                
                all_time_list = []
                
                for i in range(0, len(result_dict[para][workload])):
                    tmp_result_dict = dict()
                    for result in result_dict[para][workload][config][i]:
                        tmp_result_dict[result] = result_dict[para][workload][config][i][result]
                
                    overall_time = 0
                    for result in tmp_result_dict:
                        overall_time += tmp_result_dict[result]
                
                    if col_index == 0:
                        worksheet.write('A' + str(row_index), 'Time')
                
                    worksheet.write(current_col + str(row_index), overall_time)
                    all_time_list.append(overall_time)
                
                    row_index += 1
                
                if para not in std_dict:
                    std_dict[para] = dict()
                if workload not in std_dict[para]:
                    std_dict[para][workload] = dict()
                
                std_dict[para][workload][config] = np.std(all_time_list) / np.mean(all_time_list)               
                col_index += 1
                
    workbook.close()
    
    avg_std_dict = dict()
    mean_avg_std_dict = dict()
    workload_list = []
    parameter_list = []
    for para in std_dict:
        avg_std_dict[para] = dict()
        overall_std_list = []
        for workload in std_dict[para]:
            overall_std = 0
            for config in std_dict[para][workload]:
                overall_std += std_dict[para][workload][config] / len(config_list)
            avg_std_dict[para][workload] = overall_std
            overall_std_list.append(overall_std)
            # avg_std_dict[para][workload] = std_dict[para][workload]['standard']
        sorted(avg_std_dict[para])
        workload_list = dict_to_list(avg_std_dict[para])
        
        mean_avg_std_dict[para] = gmean(overall_std_list)
    sorted(avg_std_dict)
    parameter_list = dict_to_list(avg_std_dict)
    
    
    
    avg_std_csv_file = output_file.replace(".xlsx", '') + '_std.csv'
    out = open(avg_std_csv_file, "w")
    
    out.write('group,')
    for i in range(0, len(parameter_list)):
        out.write(parameter_list[i])
        if i != len(parameter_list) - 1:
            out.write(',')
        else:
            out.write(os.linesep)
            
    for i in range(0, len(workload_list)):
        out.write(workload_list[i]+',')
        for j in range(0, len(parameter_list)):
            out.write(str(avg_std_dict[parameter_list[j]][workload_list[i]]))
            if j != len(parameter_list) - 1:
                out.write(',')
            else:
                out.write(os.linesep)
                
    out.write('Geo-mean,')
    for j in range(0, len(parameter_list)):
        out.write(str(mean_avg_std_dict[parameter_list[j]]))
        if j != len(parameter_list) - 1:
            out.write(',')
        else:
            out.write(os.linesep)
    out.close()



    vector_seq_mega_csv_file = gpu + '_vector_seq_maga.csv'
    if 'mega' in parameter_super_list:
        out = open(vector_seq_mega_csv_file, "w")
        
        out.write('group,,Time')
        out.write(os.linesep)
                
        for i in range(0, len(result_dict['mega']['vector_seq']['standard'])):
            gpu_kernel = result_dict['mega']['vector_seq']['standard'][i]['gpu_kernel']
            memcpy = result_dict['mega']['vector_seq']['standard'][i]['memcpy']
            allocation = result_dict['mega']['vector_seq']['standard'][i]['allocation']

            out.write(str(i)+','+'gpu_kernel'+',')
            out.write(str(gpu_kernel))
            out.write(os.linesep)
            out.write(str(i)+','+'allocation'+',')
            out.write(str(allocation))
            out.write(os.linesep)
            out.write(str(i)+','+'memcpy'+',')
            out.write(str(memcpy))
            out.write(os.linesep)
        out.close()


    mega_avg_csv_file = gpu + '_mega_avg.csv'
    if 'mega' in parameter_super_list:
        mega_avg_dict = dict()
        avg = dict()
        for c in range(0, len(config_list)):
            avg[config_list[c]] = dict()

            avg[config_list[c]]['gpu_kernel'] = 0
            avg[config_list[c]]['memcpy'] = 0
            avg[config_list[c]]['allocation'] = 0
        for workload in workload_list:
            mega_avg_dict[workload] = dict()
            for c in range(0, len(config_list)):
                mega_avg_dict[workload][config_list[c]] = dict()
                mega_avg_dict[workload][config_list[c]]['gpu_kernel'] = 0
                mega_avg_dict[workload][config_list[c]]['memcpy'] = 0
                mega_avg_dict[workload][config_list[c]]['allocation'] = 0
                mega_avg_dict[workload][config_list[c]]['all'] = 0
                for i in range(0, len(result_dict['mega'][workload][config_list[c]])):
                    mega_avg_dict[workload][config_list[c]]['gpu_kernel'] += result_dict['mega'][workload][config_list[c]][i]['gpu_kernel']
                    mega_avg_dict[workload][config_list[c]]['memcpy'] += result_dict['mega'][workload][config_list[c]][i]['memcpy']
                    mega_avg_dict[workload][config_list[c]]['allocation'] += result_dict['mega'][workload][config_list[c]][i]['allocation']
                    
                    mega_avg_dict[workload][config_list[c]]['all'] += result_dict['mega'][workload][config_list[c]][i]['gpu_kernel']
                    mega_avg_dict[workload][config_list[c]]['all'] += result_dict['mega'][workload][config_list[c]][i]['memcpy']
                    mega_avg_dict[workload][config_list[c]]['all'] += result_dict['mega'][workload][config_list[c]][i]['allocation']
            for c in range(0, len(config_list)):
                print(workload, config_list[c])
                # print(para, workload, config, i, overall_time)
                normarlized_all = mega_avg_dict[workload][config_list[c]]['all'] / mega_avg_dict[workload]['standard']['all']
                mega_avg_dict[workload][config_list[c]]['gpu_kernel'] = (mega_avg_dict[workload][config_list[c]]['gpu_kernel'] / mega_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                mega_avg_dict[workload][config_list[c]]['memcpy'] = (mega_avg_dict[workload][config_list[c]]['memcpy'] / mega_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                mega_avg_dict[workload][config_list[c]]['allocation'] = (mega_avg_dict[workload][config_list[c]]['allocation'] / mega_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                
                avg[config_list[c]]['gpu_kernel'] += mega_avg_dict[workload][config_list[c]]['gpu_kernel'] / len(workload_list)
                avg[config_list[c]]['memcpy'] += mega_avg_dict[workload][config_list[c]]['memcpy'] / len(workload_list)
                avg[config_list[c]]['allocation'] += mega_avg_dict[workload][config_list[c]]['allocation'] / len(workload_list)
        sorted(mega_avg_dict)
        mega_avg_dict['AVG'] = avg
        export_workload_list = workload_list + ['AVG']
        out = open(mega_avg_csv_file, "w")

        out.write('group,,')
        for i in range(0, len(config_list)):
            out.write(config_list[i].replace('_prefetch', '').replace('pinned', 'pm'))
            if i != len(config_list) - 1:
                out.write(',')
            else:
                out.write(os.linesep)

        for i in range(0, len(export_workload_list)):
            
            out.write(export_workload_list[i]+',gpu_kernel,')
            for j in range(0, len(config_list)):
                out.write(str(mega_avg_dict[export_workload_list[i]][config_list[j]]['gpu_kernel']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',memcpy,')
            for j in range(0, len(config_list)):
                out.write(str(mega_avg_dict[export_workload_list[i]][config_list[j]]['memcpy']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',allocation,')
            for j in range(0, len(config_list)):
                out.write(str(mega_avg_dict[export_workload_list[i]][config_list[j]]['allocation']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
        out.close()
    
    super_avg_csv_file = gpu + '_super_avg.csv'
    if 'super' in parameter_super_list:
        super_avg_dict = dict()
        avg = dict()
        for c in range(0, len(config_list)):
            avg[config_list[c]] = dict()

            avg[config_list[c]]['gpu_kernel'] = 0
            avg[config_list[c]]['memcpy'] = 0
            avg[config_list[c]]['allocation'] = 0
        for workload in workload_list:
            super_avg_dict[workload] = dict()
            for c in range(0, len(config_list)):
                super_avg_dict[workload][config_list[c]] = dict()
                super_avg_dict[workload][config_list[c]]['gpu_kernel'] = 0
                super_avg_dict[workload][config_list[c]]['memcpy'] = 0
                super_avg_dict[workload][config_list[c]]['allocation'] = 0
                super_avg_dict[workload][config_list[c]]['all'] = 0
                for i in range(0, len(result_dict['super'][workload][config_list[c]])):
                    super_avg_dict[workload][config_list[c]]['gpu_kernel'] += result_dict['super'][workload][config_list[c]][i]['gpu_kernel']
                    super_avg_dict[workload][config_list[c]]['memcpy'] += result_dict['super'][workload][config_list[c]][i]['memcpy']
                    super_avg_dict[workload][config_list[c]]['allocation'] += result_dict['super'][workload][config_list[c]][i]['allocation']
                    print(i, config_list[c], result_dict['super'][workload][config_list[c]][i]['memcpy'])

                    super_avg_dict[workload][config_list[c]]['all'] += result_dict['super'][workload][config_list[c]][i]['gpu_kernel']
                    super_avg_dict[workload][config_list[c]]['all'] += result_dict['super'][workload][config_list[c]][i]['memcpy']
                    super_avg_dict[workload][config_list[c]]['all'] += result_dict['super'][workload][config_list[c]][i]['allocation']
            
            for c in range(0, len(config_list)):
                normarlized_all = super_avg_dict[workload][config_list[c]]['all'] / super_avg_dict[workload]['standard']['all']
                super_avg_dict[workload][config_list[c]]['gpu_kernel'] = (super_avg_dict[workload][config_list[c]]['gpu_kernel'] / super_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                super_avg_dict[workload][config_list[c]]['memcpy'] = (super_avg_dict[workload][config_list[c]]['memcpy'] / super_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                super_avg_dict[workload][config_list[c]]['allocation'] = (super_avg_dict[workload][config_list[c]]['allocation'] / super_avg_dict[workload][config_list[c]]['all']) * normarlized_all

                avg[config_list[c]]['gpu_kernel'] += super_avg_dict[workload][config_list[c]]['gpu_kernel'] / len(workload_list)
                avg[config_list[c]]['memcpy'] += super_avg_dict[workload][config_list[c]]['memcpy'] / len(workload_list)
                avg[config_list[c]]['allocation'] += super_avg_dict[workload][config_list[c]]['allocation'] / len(workload_list)
        sorted(super_avg_dict)
        super_avg_dict['AVG'] = avg
        export_workload_list = workload_list + ['AVG']
        out = open(super_avg_csv_file, "w")

        out.write('group,,')
        for i in range(0, len(config_list)):
            out.write(config_list[i].replace('_prefetch', '').replace('pinned', 'pm'))
            if i != len(config_list) - 1:
                out.write(',')
            else:
                out.write(os.linesep)

        for i in range(0, len(export_workload_list)):
            
            out.write(export_workload_list[i]+',gpu_kernel,')
            for j in range(0, len(config_list)):
                out.write(str(super_avg_dict[export_workload_list[i]][config_list[j]]['gpu_kernel']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',memcpy,')
            for j in range(0, len(config_list)):
                out.write(str(super_avg_dict[export_workload_list[i]][config_list[j]]['memcpy']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',allocation,')
            for j in range(0, len(config_list)):
                out.write(str(super_avg_dict[export_workload_list[i]][config_list[j]]['allocation']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
        out.close()
    
    large_avg_csv_file = gpu + '_large_avg.csv'
    if 'large' in parameter_super_list:
        large_avg_dict = dict()
        avg = dict()
        for c in range(0, len(config_list)):
            avg[config_list[c]] = dict()

            avg[config_list[c]]['gpu_kernel'] = 0
            avg[config_list[c]]['memcpy'] = 0
            avg[config_list[c]]['allocation'] = 0
        for workload in workload_list:
            large_avg_dict[workload] = dict()
            for c in range(0, len(config_list)):
                large_avg_dict[workload][config_list[c]] = dict()
                large_avg_dict[workload][config_list[c]]['gpu_kernel'] = 0
                large_avg_dict[workload][config_list[c]]['memcpy'] = 0
                large_avg_dict[workload][config_list[c]]['allocation'] = 0
                large_avg_dict[workload][config_list[c]]['all'] = 0
                for i in range(0, len(result_dict['large'][workload][config_list[c]])):
                    large_avg_dict[workload][config_list[c]]['gpu_kernel'] += result_dict['large'][workload][config_list[c]][i]['gpu_kernel']
                    large_avg_dict[workload][config_list[c]]['memcpy'] += result_dict['large'][workload][config_list[c]][i]['memcpy']
                    large_avg_dict[workload][config_list[c]]['allocation'] += result_dict['large'][workload][config_list[c]][i]['allocation']
                    
                    large_avg_dict[workload][config_list[c]]['all'] += result_dict['large'][workload][config_list[c]][i]['gpu_kernel']
                    large_avg_dict[workload][config_list[c]]['all'] += result_dict['large'][workload][config_list[c]][i]['memcpy']
                    large_avg_dict[workload][config_list[c]]['all'] += result_dict['large'][workload][config_list[c]][i]['allocation']
            
            for c in range(0, len(config_list)):
                normarlized_all = large_avg_dict[workload][config_list[c]]['all'] / large_avg_dict[workload]['standard']['all']
                large_avg_dict[workload][config_list[c]]['gpu_kernel'] = (large_avg_dict[workload][config_list[c]]['gpu_kernel'] / large_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                large_avg_dict[workload][config_list[c]]['memcpy'] = (large_avg_dict[workload][config_list[c]]['memcpy'] / large_avg_dict[workload][config_list[c]]['all']) * normarlized_all
                large_avg_dict[workload][config_list[c]]['allocation'] = (large_avg_dict[workload][config_list[c]]['allocation'] / large_avg_dict[workload][config_list[c]]['all']) * normarlized_all
        
                avg[config_list[c]]['gpu_kernel'] += large_avg_dict[workload][config_list[c]]['gpu_kernel'] / len(workload_list)
                avg[config_list[c]]['memcpy'] += large_avg_dict[workload][config_list[c]]['memcpy'] / len(workload_list)
                avg[config_list[c]]['allocation'] += large_avg_dict[workload][config_list[c]]['allocation'] / len(workload_list)
        sorted(large_avg_dict)
        large_avg_dict['AVG'] = avg
        export_workload_list = workload_list + ['AVG']
        out = open(large_avg_csv_file, "w")

        out.write('group,,')
        for i in range(0, len(config_list)):
            out.write(config_list[i].replace('_prefetch', '').replace('pinned', 'pm'))
            if i != len(config_list) - 1:
                out.write(',')
            else:
                out.write(os.linesep)

        for i in range(0, len(export_workload_list)):
            
            out.write(export_workload_list[i]+',gpu_kernel,')
            for j in range(0, len(config_list)):
                out.write(str(large_avg_dict[export_workload_list[i]][config_list[j]]['gpu_kernel']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',memcpy,')
            for j in range(0, len(config_list)):
                out.write(str(large_avg_dict[export_workload_list[i]][config_list[j]]['memcpy']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
            out.write(export_workload_list[i]+',allocation,')
            for j in range(0, len(config_list)):
                out.write(str(large_avg_dict[export_workload_list[i]][config_list[j]]['allocation']))
                if j != len(config_list) - 1:
                    out.write(',')
                else:
                    out.write(os.linesep)
                    
        out.close()

    return avg_std_csv_file, vector_seq_mega_csv_file, mega_avg_csv_file, super_avg_csv_file, large_avg_csv_file
    

def plot_breakdown_results(csv_file, output_file):
    df = pd.read_csv(csv_file, index_col=[0,1])
    group_list = []
    subgrp_list = []
    for index in df.index:
        if index[0] not in group_list:
            group_list.append(index[0])
        if index[1] not in subgrp_list:
            subgrp_list.append(index[1])
    col_list = df.columns

    ngroups = len(group_list)
    nsubgrps = len(subgrp_list)
    x = np.arange(ngroups)
    nbars = len(col_list)
    width = (1 - 0.4) / (1.5 * nbars) # the width of the bars

    matplotlib.rcParams["hatch.linewidth"] = 2

    fig, ax = plt.subplots(figsize=[8.8, 2.8])
    hdl_pair = []

    rects = []

    for i in range(0, nbars):
        height_cum = np.array([0.0] * ngroups)
        height_total = np.array([df[col_list[i]][g][0] + df[col_list[i]][g][1] + df[col_list[i]][g][2] for g in group_list])  # y coo
        height_curr = np.array([df[col_list[i]][g][0] for g in group_list])  # y coo
        rect_base = ax.bar(x - 0.3 + (3 * i + 1.5) * width / 2,          # x coo
                        height_curr,  # y coo
                        width, label=subgrp_list[0].replace('standard', 'baseline'),
                        color=color_tab[i],
                        edgecolor=color_tab[i],
                        linewidth=0.5
                        )
                        # hatch=patterns[i]*3
        rects.append(rect_base)
        height_cum += height_curr
        for j in range(1, 3):
            height_curr = np.array([df[col_list[i]][g][j] for g in group_list])
            rect = ax.bar(x - 0.3 + (3 * i + 1.5) * width / 2,          # x coo
                        height_curr,  # y coo
                        width,
                        bottom=height_cum,
                        label=subgrp_list[j].replace('standard', 'baseline'),
                        color=color_tab[i],
                        edgecolor=color_tab[i],
                        linewidth=0.5,
                        alpha=0.25 * (4 - j)
                        )
            rects.append(rect)
            height_cum += height_curr

    hdl_pair = [(rects[i*nsubgrps], rects[i*nsubgrps+1]) for i in range(nbars)]
    ax.set_xticks(x)
    ax.set_xticklabels(group_list, rotation = 0)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
    ax.legend(fontsize=14)
    # ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # ax.set_yscale('log')
    
    plt.xticks(fontsize=14, rotation=15)
    plt.yticks(fontsize=14)
    plt.grid(axis='y')
    plt.xlabel("")
    plt.ylabel("Execution Time (ns)", fontsize=14)
    plt.tight_layout()
    # plt.margins(x=0.01, y=0.01)

    plt.savefig(output_file + '_std.pdf', bbox_inches='tight')
    plt.savefig(output_file + '_std.png', bbox_inches='tight')
    plt.close()


def plot_breakdown_avg_results(csv_file, output_file):
    df = pd.read_csv(csv_file, index_col=[0, 1])
    group_list = []
    subgrp_list = []
    for index in df.index:
        if index[0] not in group_list:
            group_list.append(index[0])
        if index[1] not in subgrp_list:
            subgrp_list.append(index[1])
    col_list = df.columns

    ngroups = len(group_list)
    nsubgrps = len(subgrp_list)
    x = np.arange(ngroups)
    nbars = len(col_list)
    width = (1 - 0.25) / (nbars)  # the width of the bars

    matplotlib.rcParams["hatch.linewidth"] = 2

    fig, ax = plt.subplots(figsize=[8.8, 3.8])
    hdl_pair = []

    rects = []

    for i in range(0, nbars):
        height_cum = np.array([0.0] * ngroups)
        height_curr = np.array([df[col_list[i]][g][0]
                                for g in group_list])  # y coo
        rect_base = ax.bar(x - 0.35 + 1.2 * i * width,          # x coo
                            height_curr,  # y coo
                            width,
                            label=str(col_list[i]+" "+subgrp_list[0]).replace('standard', 'baseline'),
                            color=color_tab[i],
                            edgecolor=color_tab[0],
                            linewidth=0.25
                            )
        rects.append(rect_base)
        height_cum += height_curr
        

        height_curr = np.array([df[col_list[i]][g][1]
                                for g in group_list])
        rect = ax.bar(x - 0.35 + 1.2 * i * width,          # x coo
                        height_curr,  # y coo
                        width,
                        label=str(col_list[i]+" "+subgrp_list[1]).replace('standard', 'baseline'),
                        bottom=height_cum,
                        color='none',
                        edgecolor=color_tab[i],
                        linewidth=0.25,
                        hatch='///'
                        )
        rects.append(rect)
        rect = ax.bar(x - 0.35 + 1.2 * i * width,          # x coo
                        height_curr,  # y coo
                        width,
                        label=str(col_list[i]+" "+subgrp_list[1]).replace('standard', 'baseline'),
                        bottom=height_cum,
                        color='none',
                        edgecolor=color_tab[0],
                        linewidth=0.25,
                        )
        height_cum += height_curr
        
        
        height_curr = np.array([df[col_list[i]][g][2]
                                for g in group_list])
        rect = ax.bar(x - 0.35 + 1.2 * i * width,          # x coo
                        height_curr,  # y coo
                        width,
                        label=str(col_list[i]+" "+subgrp_list[2]).replace('standard', 'baseline'),
                        bottom=height_cum,
                        color=color_tab[i],
                        edgecolor=color_tab[0],
                        linewidth=0.25,
                        alpha = 0.25 * 2
                        )
        rects.append(rect)
        height_cum += height_curr

    hdl_pair = [(rects[i*nsubgrps], rects[i*nsubgrps+1], rects[i*nsubgrps+2]) for i in range(nbars)]
    ax.set_xticks(x)
    ax.set_xticklabels(group_list, rotation=0)

    col_list = col_list.to_list()
    for i in range(0, len(col_list)):
        col_list[i] = col_list[i].replace("standard", "baseline")
    print(col_list)
    # ax.legend(hdl_pair, col_list, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2), fontsize=14, handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0.25, labelspacing=0.25, handletextpad=0.25)
    ax.legend(hdl_pair, col_list, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.12), fontsize=14, handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0.2, labelspacing=0.2, handletextpad=0.2, columnspacing=0.5)
    # ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # ax.set_yscale('log')

    plt.xticks(fontsize=14, rotation=10)
    plt.yticks(fontsize=14)
    plt.grid(axis='y')
    plt.xlabel("")
    plt.ylabel("Time (normalized to baseline)", fontsize=14)
    plt.tight_layout()
    # plt.margins(x=0.01, y=0.01)

    plt.savefig(output_file + '_std.pdf', bbox_inches='tight')
    plt.savefig(output_file + '_std.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    addOptions(parser)
    
    options = parser.parse_args()
    
    iterations = options.iterations
    gpu = options.gpu
    output_csv_file = options.output.replace('.xlsx', '_' + options.gpu + '.xlsx')
    output_figure_file = options.figure + '_' + options.gpu
    profiling = options.profiling
    clean = options.clean
    
    root_directory = './'
    
    config_list = get_config_list(root_directory)
    workload_list, workload_dict = get_workload_dict(root_directory, config_list)
    
    if clean:
        execute_clean_bashes(workload_dict)
    else:
        if profiling:
            execute_bashes(workload_dict, gpu, iterations)

        root_directory = '../../traces/micro/'
        
        config_list = get_config_list(root_directory)
        config_list = config_super_list
        workload_list, workload_dict = get_workload_dict(root_directory, config_list)
        print(workload_dict)
        
        result_dict = process_results(workload_dict, gpu, iterations)
        export_xlsx(result_dict, config_list, iterations, output_csv_file)
        plot_results(result_dict, config_list, workload_list, iterations, output_figure_file)
        
        avg_std_csv_file, vector_seq_mega_csv_file, mega_avg_csv_file, super_avg_csv_file, large_avg_csv_file = export_xlsx_all(result_dict, gpu, config_super_list, iterations, output_csv_file)
        
        mega_avg_csv_file = gpu + '_mega_avg.csv'
        super_avg_csv_file = gpu + '_super_avg.csv'
        large_avg_csv_file = gpu + '_large_avg.csv'

        if 'large' in parameter_super_list:
            plot_breakdown_avg_results(large_avg_csv_file, 'micro_large_avg' + '_' + options.gpu)
        if 'super' in parameter_super_list:
            plot_breakdown_avg_results(super_avg_csv_file, 'micro_super_avg' + '_' + options.gpu)
        if 'mega' in parameter_super_list:
            plot_breakdown_avg_results(mega_avg_csv_file, 'micro_mega_avg' + '_' + options.gpu)
            # plot_breakdown_results(vector_seq_mega_csv_file, 'vector_seq_mega' + '_' + options.gpu)


if __name__ == '__main__':
    main()

