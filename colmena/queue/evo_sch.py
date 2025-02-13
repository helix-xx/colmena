# Standard library imports
from email import message
import re
import uuid
import copy
import gc
import json
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import time
import datetime
import heapq
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, ClassVar, Collection, Dict, List, Literal, Optional, Union
import bisect
from numba import jit, float64, int32

from functools import lru_cache
from xml.sax.handler import all_features
# Third-party library imports
import numpy as np
import pandas as pd
from pandas import DataFrame
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import fill_

# Lazy imports for heavy ML modules
# def get_ml_models():
#     from sklearn.linear_model import Ridge, ElasticNet
#     return Ridge, ElasticNet

# Local application imports
from colmena.models import Result

# Configure logging
import logging
logging.getLogger("sklearnex").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Path configuration
def setup_path():
    relative_path = "~/project/colmena/multisite_"
    absolute_path = os.path.expanduser(relative_path)
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)

setup_path()

from my_util.data_structure import *

# Define module exports
# __all__ = [
#     'SmartScheduler',
#     'agent_pilot',
#     'individual',
#     'available_task',
#     'Sch_data',
#     'evosch2',
#     'FCFSScheduler'
# ]


def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj


class SmartScheduler:
    # support different scheduling policy here
    
    ## init all sch model here
    # sch_data can be menber of all member model
    def __init__(self, methods, available_task_capacity, available_resources, sch_config= None):
        self.sch_data: Sch_data = Sch_data(methods)
        # self.agent_pilot = agent_pilot(sch_data=self.sch_data, resources_rate=2, available_resources=available_resources, util_level=0.8)
        self.sch_data.init_task_queue(available_task(methods), available_task_capacity)
        self.sch_data.init_hist_task(HistoricalData(methods))
        self.sch_data.init_task_time_predictor(methods, self.sch_data.historical_task_data.features)
        self.evo_sch: evosch2 = evosch2(resources=available_resources, at=self.sch_data.avail_task, hist_data=self.sch_data.historical_task_data, sch_data=self.sch_data)
        self.fcfs_sch: FCFSScheduler = FCFSScheduler(resources=available_resources, at=self.sch_data.avail_task, hist_data=self.sch_data.historical_task_data, sch_data=self.sch_data)
        
        #agent_pilot
        self.resources_rate = 2
        self.available_resources = available_resources
        self.record_resources = copy.deepcopy(available_resources)
        self.util_level = 0.1
        self.exceed_area_limit = 1.1
        self.exceed_completion_time_limit = 1
        
        
        self.best_result = None
        
        # lock
        self.sch_lock = threading.Lock()
        self.available_task_lock = threading.Lock() # lock for available task to move task between available and scheduled
        
        # processes = len(self.node_resources)
        processes = 4
        self.pool = multiprocessing.Pool(processes=processes)
        
        self.sch_data.usr_path = os.path.expanduser('~')
        

        # init historical data and task time predictor
        hist_path = []
        # hist_path on Research and teaching cluster
        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/test_data/simulation-results-20241224-116.json')
        # )
        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/test_data/simulation-results-20241224-152.json')
        # )

        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/inference-results-20240319_230707.json')
        # )
        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/sampling-results-20241211.json')
        # )
        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/training-results-20241211.json')
        # )
        
        # hist_path on qiming
        hist_path.append(
            os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/qimingdata/simulation-results.json')
        )
        hist_path.append(
            os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/qimingdata/training-results.json')
        )
        self.sch_data.historical_task_data.get_features_from_his_json(hist_path)
        self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
        self.sch_data.Task_time_predictor.fill_time_running_database(self.available_resources, self.sch_data.historical_task_data.historical_data)
        # self.sch_data.Task_time_predictor.fill_features_from_new_task(self.available_resources, self.sch_data.sch_task_list)
        self.sch_data.Task_time_predictor.fill_runtime_records_with_predictor()
        
        logger.info('init smart scheduler')
        
        # warmup numba functions
        self._warmup_numba_functions()
        
    def __del__(self):
        self.pool.close()
        self.pool.join()
        
    def _warmup_numba_functions(self):
        """预热所有numba函数"""
        print("Warming up numba functions...")
        start = time.time()
        
        # 准备最小规模的测试数据
        small_task_cpu = np.array([1], dtype=np.int32)
        small_task_gpu = np.array([0], dtype=np.int32)
        small_task_runtime = np.array([1.0], dtype=np.float64)
        empty_running = np.array([], dtype=np.float64)
        empty_cpus = np.array([], dtype=np.int32)
        empty_gpus = np.array([], dtype=np.int32)
        
        # 预热计算完成时间函数
        _calculate_completion_time(
            small_task_cpu,
            small_task_gpu,
            small_task_runtime,
            empty_running,
            empty_cpus,
            empty_gpus,
            4,
            2,
            0.0
        )
        print(f"Warmed up in {time.time() - start:.2f} seconds")
        
    
    def acquire_resources(self, key):
        # topic与method不一样，暂时添加一个映射
        topic_method_mapping = {
            'simulate': 'run_calculator',
            'sample': 'run_sampling',
            'train': 'train',
            'infer': 'evaluate'
        }
        
        method = topic_method_mapping.get(key, None)  # 根据给定的 key 获取对应的方法
        
        pilot_task = self.sch_data.pilot_task.get(method, None)
        info = {}
        if not pilot_task:
            info['reason'] = "no pilot task"
            return 0, info
        else:
            # 获取前后分配的情况，并通过预测器的时间计算资源利用率
            ind = self.best_result
            if not ind:
                info['reason'] = 'no previous info'
                return 0, info
            total_cpu_time_per_node, total_gpu_time_per_node, completion_time = self.evo_sch.calc_utilization(ind)
            
            all_tasks = self.sch_data.avail_task.get_all()
            all_tasks = copy.deepcopy(all_tasks)
            pilot_task = copy.deepcopy(pilot_task)
            self.sch_data.renew_task_uuid(pilot_task)
            
            all_tasks = self.sch_data.avail_task.dummy_add_task_id(method, pilot_task['task_id'], all_tasks=all_tasks)
            self.sch_data.add_sch_task(pilot_task)
            _ = self.run_sch()
            new_ind = self.evo_sch.best_ind
            new_total_cpu_time_per_node, new_total_gpu_time_per_node, new_completion_time = self.evo_sch.calc_utilization(new_ind)
            self.sch_data.pop_sch_task(pilot_task)
            
            # evaluate resources
            node_cpu_count = {node: self.available_resources[node]['cpu'] for node in self.available_resources.keys()}
            node_gpu_count = {node: self.available_resources[node]['gpu'] for node in self.available_resources.keys()}

            current_cpu_utilization = {node: (total_cpu_time_per_node[node] / (completion_time[node] * node_cpu_count[node]) if completion_time[node] > 0 else 0) for node in total_cpu_time_per_node}
            new_cpu_utilization = {node: (new_total_cpu_time_per_node[node] / (new_completion_time[node] * node_cpu_count[node]) if new_completion_time[node] > 0 else 0) for node in new_total_cpu_time_per_node}
            
            current_gpu_utilization = {node: (total_gpu_time_per_node[node] / (completion_time[node] * node_gpu_count[node]) if completion_time[node] > 0 else 0) for node in total_gpu_time_per_node}
            new_gpu_utilization = {node: (new_total_gpu_time_per_node[node] / (new_completion_time[node] * node_gpu_count[node]) if new_completion_time[node] > 0 else 0) for node in new_total_gpu_time_per_node}
            
            # 计算利用率提升比例
            utilization_improvement = {}
            for node in current_cpu_utilization:
                cpu_improvement_ratio = (
                    (new_cpu_utilization[node] - current_cpu_utilization[node]) / current_cpu_utilization[node]
                    if current_cpu_utilization[node] > 0 else float('inf')
                )
                gpu_improvement_ratio = (
                    (new_gpu_utilization[node] - current_gpu_utilization[node]) / current_gpu_utilization[node]
                    if current_gpu_utilization[node] > 0 else float('inf')
                )
                
                utilization_improvement[node] = {
                    'cpu': cpu_improvement_ratio,
                    'gpu': gpu_improvement_ratio
                }
                
            # 计算计算时间的延长
            completion_time_improvement = {node: (new_completion_time[node] - completion_time[node]) / completion_time[node]  if completion_time[node] > 0 else 0 for node in completion_time}
            
            used_cpu_area = sum(total_cpu_time_per_node.values())
            new_used_cpu_area = sum(new_total_cpu_time_per_node.values())
            used_gpu_area = sum(total_gpu_time_per_node.values())
            new_used_gpu_area = sum(new_total_gpu_time_per_node.values())
            
            cpu_area_per_node = {}
            new_cpu_area_per_node = {}
            gpu_area_per_node = {}
            new_gpu_area_per_node = {}

            # completion time with cpu and gpu weight
            for node in completion_time:
                cpu_nums = self.available_resources[node]['cpu']
                cpu_area_per_node[node] = completion_time[node] * cpu_nums
                new_cpu_area_per_node[node] = new_completion_time[node] * cpu_nums
                gpu_nums = self.available_resources[node]['gpu']
                gpu_area_per_node[node] = completion_time[node] * gpu_nums
                new_gpu_area_per_node[node] = new_completion_time[node] * gpu_nums
            
            total_cpu_area = sum(cpu_area_per_node.values())
            new_total_cpu_area = sum(new_cpu_area_per_node.values())
            total_gpu_area = sum(gpu_area_per_node.values())
            new_total_gpu_area = sum(new_gpu_area_per_node.values())
            
            # total util improvement
            # TODO area 可能不变，用标准任务 / area获得潜在效率提升
            # 检查占用总area是否超出：
            if new_total_cpu_area > self.exceed_area_limit * total_cpu_area or new_total_gpu_area > self.exceed_area_limit * total_gpu_area:
                info['reason'] = "exceed area limit"
                return 0, info
            # 检查最大 completion time是否超出
            if max(new_completion_time.values()) > self.exceed_completion_time_limit * max(completion_time.values()):
                info['reason'] = "exceed completion time limit"
                return 0, info
            
            # 检查是否有提升达到设定的 util_level
            # 设置的提升阈值（例如10%）
            info = utilization_improvement
            for node, improvement in utilization_improvement.items():
                if improvement['cpu'] >= self.util_level or improvement['gpu'] >= self.util_level:
                    info['reason'] = "utilize improvement;"
                    return 1, info
            
            # info['reason'] = "no utilize improvement"
            # logger.info('acquire resources info {}'.format(info))
            # return 0, info
            info['reason'] = "no limit exceed"
            return 1, info
        
    def run_sch(self, method = "ga", model_type="powSum"):
        """运行调度器

        Args:
            method (str, optional): 选择ga或mrsa调度器. Defaults to "ga".
            model_type (str, optional): 选择mrsa时设置amdMax, amdSum, powMax, powSum四种性能模型. Defaults to "powSum".

        Returns:
            _type_: _description_
        """
        if method == "ga":
        # run evo sch
            # with self.sch_lock: # 异步进行不需要加锁，每个调度算法都有自己的可调度任务
            all_tasks = self.sch_data.avail_task.get_all()
            self.sch_data.avail_task.move_available_to_scheduled(all_tasks) # 线程安全
            best_allocation = self.evo_sch.run_ga(all_tasks, pool = self.pool)
            self.sch_data.avail_task.move_allocation_to_scheduled(best_allocation) # 线程安全
            self.best_result = self.evo_sch.best_ind
            return best_allocation
        elif method == "mrsa":
            with self.sch_lock:
                self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
                best_allocation = self.run_mrsa_scheduler(model_type=model_type)
                return best_allocation

    def run_mrsa_scheduler(self, model_type="powSum"):
        """使用MRSA替代GA进行调度"""
        # 准备输入文件
        folder_name = prepare_mrsa_input(self.sch_data, model_type)
        folder_name = "fitune_surrogate"
        
        # 获取资源配置
        # total_cpu = sum(node['cpu'] for node in self.available_resources.values())
        # total_gpu = sum(node['gpu'] for node in self.available_resources.values())
        first_node = list(self.available_resources.keys())[0]
        total_cpu = self.available_resources[first_node]['cpu']
        total_gpu = self.available_resources[first_node]['gpu']
        
        # 调用MRSA调度器
        mrsa_path = self.sch_data.usr_path + "/project/colmena/multisite_/mrsa"
        cmd = f"python3 {mrsa_path}/alphaMaster.py {folder_name} 2 {total_cpu} {total_gpu} {model_type} {folder_name}"
        logger.info(f'Running MRSA scheduler with command: {cmd}')
        print(cmd)
        # os.system(cmd)
        try:
            # 使用 subprocess.run 执行命令
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
            
            # 打印输出
            logger.info(result.stdout)

        except subprocess.CalledProcessError as e:
            logger.error(f'Command "{cmd}" failed with return code {e.returncode}')
            logger.error(f'Standard output: {e.stdout}')
            logger.error(f'Standard error: {e.stderr}')
        
        # 读取调度结果
        task_allocation = parse_mrsa_output(folder_name, self.sch_data, self.available_resources)
        logger.info(f'MRSA scheduler finished with {len(task_allocation)} tasks allocated')
        
        return task_allocation

## mrsa 调度
def convert_to_mrsa_models(task_list, models, model_type, output_folder="fitune_surrogate", dimensions=2):
    """将机器学习模型转换为MRSA支持的四种性能模型
    
    Args:
        task_list: 任务列表
        models: 训练好的ML模型（多项式或随机森林）
        model_type: 'amdSum'|'amdMax'|'powSum'|'powMax'
        output_folder: 输出目录
        dimensions: 资源维度数
    """
    def estimate_model_params(task, ml_model, model_type):
        """估计单个任务的模型参数"""
        # 采样点设置
        cpu_points = np.linspace(1, 32, 10)
        gpu_points = np.linspace(1, 4, 5)   
        
        base_cpu = 1
        base_gpu = 0
        if task.get('resources.gpu', 0) >= 1:
            base_gpu = 1
        # 1. 估计串行部分 s0
        X_serial = DataFrame([{
            'message_sizes.inputs': task.get('message_sizes.inputs', 1),
            'resources.cpu': base_cpu,
            'resources.gpu': base_gpu
        }])
        serial_time = max(1, np.expm1(ml_model.predict(X_serial)[0]))
        
        # 2. 分别采样CPU和GPU的影响
        times_cpu = []  # 仅CPU配置
        times_gpu = []  # CPU+GPU配置
        configs_cpu = []
        configs_gpu = []
        
        logger.info(f'Estimating model parameters for task {task["task_id"]}, serial time: {serial_time}')
        # CPU采样
        for cpu in cpu_points:
            X = DataFrame([{
                'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                'resources.cpu': cpu,
                'resources.gpu': base_gpu
            }])
            time = max(1, np.expm1(ml_model.predict(X)[0]))
            logger.info(f'CPU sample: CPU={cpu}, time={time}, features={X.to_dict(orient="records")}')
            times_cpu.append(time)
            configs_cpu.append([cpu])
        
        # GPU采样（固定最优CPU）如果原任务没有GPU，则不进行GPU采样
        if base_gpu != 0:
            optimal_cpu = cpu_points[np.argmin(times_cpu)]
            for gpu in gpu_points[1:]:  # 跳过gpu=0
                X = DataFrame([{
                    'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                    'resources.cpu': optimal_cpu,
                    'resources.gpu': gpu
                }])
                time = max(1, np.expm1(ml_model.predict(X)[0]))
                logger.info(f'GPU sample: CPU={cpu}, time={time}, features={X.to_dict(orient="records")}')
                times_gpu.append(time)
                configs_gpu.append([optimal_cpu, gpu])
            
        # 3. 根据不同模型类型估计参数
        params = {}
        # 估计串行比例
        s0 = min(times_cpu)  # 基础串行时间
        
        # CPU部分
        X_cpu = np.array([1/c for c in cpu_points]).reshape(-1, 1)
        y_cpu = np.array(times_cpu) - s0
        reg_cpu = LinearRegression()
        reg_cpu.fit(X_cpu, y_cpu)
        s1 = max(0.1, reg_cpu.coef_[0])  # CPU并行部分权重
        
        # GPU部分
        if len(times_gpu) > 0:
            X_gpu = np.array([1/g for g in gpu_points[1:]]).reshape(-1, 1)
            # y_gpu = np.array(times_gpu) - s0 - s1/optimal_cpu
            y_gpu = np.array(times_gpu) - s0
            reg_gpu = LinearRegression()
            reg_gpu.fit(X_gpu, y_gpu)
            s2 = max(0, reg_gpu.coef_[0])  # GPU并行部分权重
        else:
            s2 = 0
            
        params = {
            's0': s0,
            's1': s1,
            's2': s2,
            'a1': 1.0,  # Amdahl模型固定为1
            'a2': 1.0
        }
        if model_type.startswith('amd'):  # Amdahl模型
            return params
            
        else:  # Power模型
            s0_amdahl = max(0, reg_cpu.intercept_)
            serial_time = min(s0-2, s0_amdahl) # 避免0或-inf
            # CPU部分
            X_cpu = np.log([c for c in cpu_points]).reshape(-1, 1)
            y_cpu = np.log(np.array(times_cpu) - serial_time)
            reg_cpu = LinearRegression()
            reg_cpu.fit(X_cpu, y_cpu)
            s1 = np.exp(reg_cpu.intercept_)
            a1 = -reg_cpu.coef_[0]
            
            # GPU部分
            if len(times_gpu) > 0:
                X_gpu = np.log(gpu_points[1:]).reshape(-1, 1)
                logger.info(f'GPU debug!! times_gpu: {times_gpu}, serial_time: {serial_time}, s1: {s1}, a1: {a1}')
                # y_gpu = np.log(np.array(times_gpu) - serial_time - s1/(optimal_cpu**a1))
                y_gpu = np.log(np.array(times_gpu) - serial_time)
                reg_gpu = LinearRegression()
                reg_gpu.fit(X_gpu, y_gpu)
                s2 = np.exp(reg_gpu.intercept_)
                a2 = -reg_gpu.coef_[0]
            else:
                s2 = 0
                a2 = 1.0
                
            params = {
                's0': serial_time,
                's1': max(0.1, s1),
                's2': max(0, s2),
                'a1': max(0.1, min(a1, 1.0)),
                'a2': max(0.1, min(a2, 1.0))
            }
            
        return params
    
    # 验证函数
    def validate_conversion(task, ml_model, params, model_type):
        """验证转换精度"""
        def predict_mrsa_time(cpu, gpu, params, model_type):
            if model_type == 'amdSum':
                return (params['s0'] + 
                       params['s1']/cpu + 
                       (params['s2']/gpu if gpu > 0 else 0))
            elif model_type == 'amdMax':
                return (params['s0'] + 
                       max(params['s1']/cpu,
                           params['s2']/gpu if gpu > 0 else 0))
            elif model_type == 'powSum':
                return (params['s0'] + 
                       params['s1']/(cpu**params['a1']) + 
                       (params['s2']/(gpu**params['a2']) if gpu > 0 else 0))
            else:  # powMax
                return (params['s0'] + 
                       max(params['s1']/(cpu**params['a1']),
                           params['s2']/(gpu**params['a2']) if gpu > 0 else 0))
        
        # 验证点
        cpu_test = np.linspace(1, 16, 8)
        gpu_test = np.linspace(0, 4, 5)
        
        errors = []
        for cpu in cpu_test:
            for gpu in gpu_test:
                # ML模型预测
                X = DataFrame([{
                    'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                    'resources.cpu': cpu,
                    'resources.gpu': gpu
                }])
                ml_time = np.expm1(ml_model.predict(X)[0])
                
                # MRSA模型预测
                mrsa_time = predict_mrsa_time(cpu, gpu, params, model_type)
                
                error = abs(ml_time - mrsa_time) / ml_time
                errors.append(error)
        
        return np.mean(errors), np.max(errors)
    
    # 主处理流程
    usr_path = os.path.expanduser('~')
    os.makedirs(f"{usr_path}/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}", exist_ok=True)
    output_file = f"{usr_path}/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}/sample0.txt"
    
    conversion_stats = {
        'mean_error': [],
        'max_error': [],
        'task_ids': []
    }
    
    with open(output_file, 'w') as f:
        for task in task_list:
            method = task['method']
            ml_model = models[method]
            
            # 估计参数
            params = estimate_model_params(task, ml_model, model_type)
            
            # 验证转换精度
            mean_err, max_err = validate_conversion(task, ml_model, params, model_type)
            conversion_stats['mean_error'].append(mean_err)
            conversion_stats['max_error'].append(max_err)
            conversion_stats['task_ids'].append(task['task_id'])
            
            # 写入MRSA格式
            line = f"{task['task_id']} {params['s0']} {params['s1']} {params['s2']} {params['a1']} {params['a2']}\n"
            f.write(line)
            
    return conversion_stats

def prepare_mrsa_input(sch_data, model_type="powSum", output_folder="fitune_surrogate"):
    """准备MRSA调度器输入"""
    # 获取当前需要调度的任务
    tasks = []
    for task_id, task in sch_data.sch_task_list.items():
        tasks.append(task)
    
    # if sch_data.Task_time_predictor.model_type == "random_forest":
    #     models = sch_data.Task_time_predictor.random_forest_models
    # elif sch_data.Task_time_predictor.model_type == "polynomial":
    #     models = sch_data.Task_time_predictor.polynomial_models
    
    models = sch_data.Task_time_predictor._models
    # 转换为power-max格式
    convert_to_mrsa_models(
        task_list=tasks,
        models=models,
        model_type=model_type,
        output_folder=output_folder
    )
    
    # 创建空的依赖关系文件夹和文件(因为不考虑依赖关系)
    usr_path = os.path.expanduser('~')
    os.makedirs(f"{usr_path}/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}", exist_ok=True)
    with open(f"{usr_path}/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}/sample0.txt", 'w') as f:
        pass
    
    # 创建分配结果文件夹
    os.makedirs(f"{sch_data.usr_path}/project/colmena/multisite_/mrsa/files/allocation/{output_folder}", exist_ok=True)
    
    return output_folder
def generate_node_cycle(nodes):
    """生成循环节点选择器"""
    node_list = list(nodes)
    index = 0
    while True:
        yield node_list[index]
        index = (index + 1) % len(node_list)
        
def parse_mrsa_output(output_folder, sch_data, node_resources):
    """解析MRSA的输出结果并转换为task_allocation格式
    
    Args:
        output_folder: MRSA输出文件夹名称
        sch_data: 调度器数据对象
        node_resources: 节点资源信息
    
    Returns:
        task_allocation: 列表形式的任务分配方案
    """
    task_allocation = []

    which_node = generate_node_cycle(node_resources.keys())
    
    # 读取MRSA的输出文件
    usr_path = os.path.expanduser('~')
    with open(f"{usr_path}/project/colmena/multisite_/mrsa/files/allocation/{output_folder}/sample0seq.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            task_id = parts[0]
            cpu_count = int(parts[1])
            gpu_count = int(parts[2])
            node1_res = node_resources['node1']
            if cpu_count > node1_res['cpu']:
                cpu_count = node1_res['cpu']
            # 获取任务的method
            task = sch_data.sch_task_list.get(task_id)
            if not task:
                continue
                
            # 简单的循环分配节点
            node = next(which_node)
            
            # 创建task_allocation格式的任务描述
            new_task = {
                "name": task['method'],
                "task_id": task_id,
                "resources": {
                    "cpu": cpu_count,
                    "gpu": gpu_count,
                    "node": node
                }
            }
            task_allocation.append(new_task)
    
    return task_allocation

# evo stargety here
@dataclass
class individual:
    # individual information
    # static variable, unique id for each individual
    _next_id: ClassVar[int] = 0
    individual_id: int = -1
    tasks_nums: int = 0
    total_resources: dict = field(default_factory=dict)
    total_time: int = 0
    max_time: int = 0
    score: int = 0

    task_array: np.ndarray = field(default=None)
    
    _task_id_index: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.individual_id == -1:
            self.individual_id = individual._next_id
            individual._next_id += 1
            

        # 定义数组的数据类型
        self.dtype = [
            ('name', 'U20'), 
            ('task_id', 'U40'),
            ('cpu', 'i4'),
            ('gpu', 'i4'),
            ('node', 'U10'),
            ('total_runtime', 'f8'),
            ('start_time', 'f8'),
            ('finish_time', 'f8')
        ]

        # 初始化空的task_array
        if self.task_array is None:
            self.task_array = np.zeros(self.tasks_nums, dtype=self.dtype)
        
        # 初始化任务索引
        self.update_task_id_index()
        
        # 添加node_array的数据类型定义
        self.node_dtype = [
            ('node', 'U10'),
            ('task_indices', 'i4', (100,))  # 假设每个节点最多100个任务
        ]
        
        # self.init_node_array()
            
    def init_node_array(self):
        """从task_array初始化node_array"""
        # 获取唯一的节点列表
        # nodes = np.unique(self.task_array['node'])
        nodes = self.total_resources.keys()
        self.node_array = np.zeros(len(nodes), dtype=self.node_dtype)
        
        # 为每个节点创建任务索引数组
        for i, node in enumerate(nodes):
            # 找到分配给该节点的所有任务
            node_mask = self.task_array['node'] == node
            node_tasks = self.task_array[node_mask]
            
            # 设置节点名称
            self.node_array[i]['node'] = node
            
            # 获取任务索引
            task_indices = [self._task_id_index[task['task_id']] 
                        for task in node_tasks]
            
            # 填充任务索引数组
            indices_len = len(task_indices)
            self.node_array[i]['task_indices'][:indices_len] = task_indices
            self.node_array[i]['task_indices'][indices_len:] = -1  # 填充-1表示无效索引
        
    @property
    def task_allocation(self):
        return [{
            'name': t['name'],
            'task_id': t['task_id'],
            'resources': {'cpu': t['cpu'], 'gpu': t['gpu'], 'node': t['node']}, 
            'total_runtime': t['total_runtime'],
            'start_time': t['start_time'],
            'finish_time': t['finish_time']
        } for t in self.task_array]

    @task_allocation.setter
    def task_allocation(self, value):
        """从字典列表格式设置task_array"""
        self._task_allocation = value  # 保存原始数据
        if not value:
            self.task_array = np.zeros(self.tasks_nums, dtype=self.dtype)
            return
            
        self.task_array = np.array(
            [(t['name'], t['task_id'], 
              t['resources']['cpu'], t['resources']['gpu'], 
              t['resources']['node'], t.get('total_runtime', 0),
              t.get('start_time', 0), t.get('finish_time', 0))
             for t in value],
            dtype=self.dtype
        )
        self.update_task_id_index()
        
    def task_array_shuffled(self):
        """随机打乱任务顺序"""
        np.random.shuffle(self.task_array)
        self.update_task_id_index()
        
    def update_task_id_index(self):
        self._task_id_index = {t['task_id']: i for i, t in enumerate(self.task_array)}
    # get task index by task_id
    def get_task_index(self, task_id):
        return self._task_id_index.get(task_id, -1)

    # deepcopy individual
    def copy(self):
        """浅拷贝+关键字段深拷贝的优化版本"""
        new_ind = individual(
            tasks_nums=self.tasks_nums, # int不可变，浅拷贝即可
            total_resources=self.total_resources,  # 字典引用（假设资源字典是只读的）
            total_time=self.total_time,
            max_time=self.max_time,
            score=self.score,
        )
        new_ind.task_array = np.copy(self.task_array)
        new_ind._task_id_index = self._task_id_index.copy()
        
        # if hasattr(self, 'node_array'):
        #     new_ind.node_array = np.copy(self.node_array)
        
        return new_ind

    # hash
    # def __hash__(self) -> int:
    #     sorted_allocation = sorted(
    #         self.task_allocation, key=lambda x: (x['name'], x['task_id'])
    #     )
    #     return hash(str(sorted_allocation))

    # def get_task_resources(self, task_name, task_id):
    #     for task in self.task_allocation:
    #         if task['name'] == task_name and task['task_id'] == task_id:
    #             return task['resources']
    #     return None

    # convert to json
    def to_json(self):
        return json.dumps(dataclass_to_dict(self), indent=4)

    # save to json file
    def save_to_json(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(dataclass_to_dict(self), f, indent=4)


class Sch_data(SingletonClass):
    def __init__(self, methods):
        self.result_list = {}
        self.sch_task_list = {}
        self.pilot_task = {}
        self.Task_time_predictor: TaskTimePredictor = None
        self.avail_task: available_task = None
        self.avail_task_cap: int = None
        self.methods = methods

    def init_hist_task(self, historical_task_data):
        self.historical_task_data:HistoricalData = historical_task_data

    def init_task_queue(self, avail_task_data, avail_task_cap):
        self.avail_task = avail_task_data
        self.avail_task_cap = avail_task_cap
        
    
    def init_task_time_predictor(self, methods, features):
        self.Task_time_predictor = TaskTimePredictor(methods, features)

    def add_result_obj(self, result: Result):
        # logger.info('add task to scheduler {}'.format(result.task_id))
        self.result_list[result.task_id] = result
        sch_task = self.historical_task_data.get_sch_task_from_result_object(result)
        logger.info(f"add sch_task: {sch_task}")
        self.sch_task_list[sch_task['task_id']] = sch_task

    def pop_result_obj(self, task_id):
        logger.info('remove task from scheduler {}'.format(task_id))
        sch_task = self.sch_task_list.pop(task_id)
        self.pilot_task[sch_task['method']] = sch_task
        return self.result_list.pop(task_id)
    
    def add_sch_task(self, sch_task):
        self.sch_task_list[sch_task['task_id']] = sch_task
        
    def pop_sch_task(self, task):
        logger.info('pop sch task {}'.format(task['task_id']))
        return self.sch_task_list.pop(task['task_id'])
    
    def renew_task_uuid(self, sch_task):
        new_task_id = uuid.uuid4()
        while str(new_task_id) in self.sch_task_list:
            new_task_id = uuid.uuid4()
        sch_task['task_id'] = str(new_task_id)

    def get_result_obj(self, task_id):
        # return self.result_list.get(task['task_id'])
        return self.result_list.get(task_id)
    
    def get_sch_task(self, task):
        return self.sch_task_list.get(task['task_id'])

    def get_result_list_len(self):
        return len(self.result_list)



@dataclass
class available_task(SingletonClass):
    # task_names: list[str] = field(default_factory=list)
    # task_ids: list[dict[str, int]] = field(default_factory=dict)

    # def __init__(self,task_names: set[str], task_ids: dict[str, int], task_datas=None):
    # def __init__(self, task_ids: dict[str, list[str]] = None):
    move_lock = threading.Lock()
    def __init__(self, task_methods: list[str]):
        self.task_ids = {method: [] for method in task_methods}
        self.scheduled_task = {method: [] for method in task_methods}
        self.allocations = []
        self.task_allocation_node = {}

    def move_allocation_to_scheduled(self, allocation):
        with self.move_lock:
            self.allocations.extend(allocation)

    def move_available_to_scheduled(self, all_tasks):
        with self.move_lock:
            for method,task_ids in all_tasks.items():
                self.scheduled_task[method].extend(task_ids)
                self.task_ids[method] = []
                
    def remove_task_from_allocation(self, task):
        with self.move_lock:
            self.allocations.remove(task)
            
    def add_task_id(self, task_name: str, task_id: Union[str, list[str]]):
        # if task_name not in self.task_names:
        #     self.task_names.add(task_name)
        task_id = [task_id] if isinstance(task_id, str) else task_id
        for i in task_id:
            self.task_ids[task_name].append(i)
    
    def dummy_add_task_id(self, task_name: str, task_id: Union[str, list[str]], all_tasks):
        # if task_name not in self.task_names:
        #     self.task_names.add(task_name)
        task_id = [task_id] if isinstance(task_id, str) else task_id
        for i in task_id:
            # self.task_ids[task_name].append(i)
            all_tasks[task_name].append(i)
        return all_tasks

    def remove_task_id(self, task_name: str, task_id: Union[str, list[str]], task_queue = 'available'):
        if task_queue == 'available':
            ## judge if there is task id
            task_id = [task_id] if isinstance(task_id, str) else task_id

            for i in task_id:
                if i not in self.task_ids[task_name]:
                    # print(f"task id {i} not in task name {task_name}")
                    # logging.warning(f"task id {task_id} not in task name {task_name}")
                    raise KeyError(f"task id {i} not in task name {task_name}")
                    continue
                self.task_ids[task_name].remove(i)
        elif task_queue == 'scheduled':
            task_id = [task_id] if isinstance(task_id, str) else task_id
            for i in task_id:
                if i not in self.scheduled_task[task_name]:
                    # print(f"task id {i} not in task name {task_name}")
                    # logging.warning(f"task id {task_id} not in task name {task_name}")
                    raise KeyError(f"task id {i} not in task name {task_name}")
                    continue
                self.scheduled_task[task_name].remove(i)
        

    def get_available_task_id(self, task_name):
        return self.task_ids.get(task_name)

    def get_all(self, task_type = 'available'):
        if task_type == 'available':
            return copy.deepcopy(self.task_ids)
        elif task_type == 'scheduled':
            return copy.deepcopy(self.scheduled_task)

    def get_task_nums(self, all_tasks):
        result = {}
        for key, value in all_tasks.items():  # key: task name, value:task id list
            result[key] = len(value)
        return result

    def get_total_nums(self, all_tasks = None, task_type = 'available'):
        if task_type == 'available':
            if all_tasks == None:
                all_tasks = self.task_ids
            return sum(self.get_task_nums(all_tasks).values())
        elif task_type == 'scheduled':
            return sum(self.get_task_nums(self.scheduled_task).values())


class HistoricalData:

    def __init__(self, methods: Collection[str], queue=None):
        self.methods = methods
        self.features = {
            "default": [
                "task_id",
                "method",
                "message_sizes.inputs",
                "resources.cpu",
                "resources.gpu",
                "time_running",
            ]
        }
        self.historical_data = {method: [] for method in methods}

    def add_feature_from_user(self, method: str, feature_values: List[str]):
        if method not in self.features:
            self.features[method] = self.features["default"][:]

        self.features[method].extend(feature_values)

    def add_data(self, feature_values: Dict[str, Any]):
        method = feature_values["method"]
        if method not in self.historical_data:
            logger.warning(f"method {method} not in historical data")
        self.historical_data[method].append(feature_values)


    def get_features_from_result_object(self, result: Result):
        feature_values = {}
        for feature in self.features['default']:
            value = result
            for key in feature.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                    break
                else:
                    value = getattr(value, key)
            if feature == 'resources.gpu':
                value = (
                    len(value) if isinstance(value, list) else value
                )  # 如果是list，则取长度，否则默认为0
                # if value is None:
                #     break
            feature_values[feature] = value
        self.add_data(feature_values)
        return feature_values
        
    def get_sch_task_from_result_object(self, result: Result):
        feature_values = {}
        for feature in self.features['default']:
            value = result
            for key in feature.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                    break
                else:
                    value = getattr(value, key)
            if feature == 'resources.gpu':
                value = (
                    len(value) if isinstance(value, list) else value
                )  # 如果是list，则取长度，否则默认为0
                # if value is None:
                #     break
            feature_values[feature] = value
        return feature_values

    def get_features_from_his_json(self, his_json: Union[str, list[str]]):
        for path in his_json:
            with open(path, 'r') as f:
                for line in f:
                    json_line = json.loads(line)
                    feature_values = {}
                    for feature in self.features['default']:
                        value = json_line
                        for key in feature.split('.'):
                            value = value.get(key)
                        if feature == 'resources.gpu':
                            value = (
                                len(value) if isinstance(value, list) else value
                            )
                            # if value is None:
                            #     break
                        feature_values[feature] = value
                    self.add_data(feature_values)    
    
class TaskTimePredictor:
    methods: List[str]
    features: Dict[str, List[str]]
    
    _models: Dict[str, Pipeline] = {}
    _feature_scalers: Dict[str, StandardScaler] = {}
    
    # 缓存训练数据的统计信息
    _feature_stats: Dict[str, Dict[str, tuple]] = {}
    
    
    
    def __init__(self, methods, features, model_for_method:Dict[str,str] = {"run_calculator": "Polynomial", "run_sampling":"random_forest","train":"random_forest","evaluate":"random_forest"}):
        """
        初始化任务时间预测器
        
        Args:
            methods: 方法列表
            features: 特征列表
            model_for_method: 每个method的模型类型，可选 "random_forest" 或 "polynomial"
        """
        self.features = features
        self.methods = methods
            
        self.time_running_db = {}
        self.dtype = np.dtype([
            ('cpu', np.int32),
            ('gpu', np.int32),
            # ('node', 'U32'),
            ('message_sizes', np.float64),
            ('time_running', np.float64)
        ])
        
        # 为每个方法初始化模型
        for method, model_type in model_for_method.items():
            self._init_model(method, model_type)
        
        # 为每个方法初始化数据库
        for method in methods:
            self.time_running_db[method] = np.array([], dtype=self.dtype)
            
        # 用于快速查找的索引
        self.method_indices = {method: {} for method in methods}

    def _create_index(self, method: str):
        """创建索引以加速查询"""
        data = self.time_running_db[method]
        if len(data) == 0:
            return
            
        # 创建资源配置到索引的映射
        self.method_indices[method] = {
            # (row['cpu'], row['gpu'], row['node']): idx 
            (row['cpu'], row['gpu']): idx
            for idx, row in enumerate(data)
        }
            
            
    def fill_time_running_database(self, node_resources: Dict[str, Dict], 
                                 historical_data: Dict[str, List[Dict]]):
        """填充运行时间数据库"""
        for method in self.methods:
            # 收集所有唯一的message_sizes值
            message_sizes = set()
            for task in historical_data[method]:
                msg_size = task.get('message_sizes.inputs')
                if msg_size is not None:
                    message_sizes.add(float(msg_size))
            # 获得节点中的最大资源
            max_cpu = max(node["cpu"] for node in node_resources.values())
            max_gpu = max(node["gpu"] for node in node_resources.values())
            
            # 生成所有可能的配置组合
            records = []
            # for node, resources in node_resources.items():
            cpu_range = range(1, max_cpu + 1)
            gpu_range = range(0, max_gpu + 1)
            
            for msg_size in message_sizes:
                for cpu in cpu_range:
                    for gpu in gpu_range:
                        records.append((
                            cpu, gpu, msg_size, np.nan
                        ))
            
            # 使用numpy结构化数组存储数据
            self.time_running_db[method] = np.array(records, dtype=self.dtype)
            self._create_index(method)

    def fill_features_from_new_task(self, node_resources: Dict[str, Dict], sch_task_list: Dict):
        """从新任务中填充特征信息并合并到现有数据库"""
        
        for method in self.methods:
            # 收集所有新的 message_sizes 值
            message_sizes = set()
            for task_id, task in sch_task_list.items():
                if task['method'] == method:
                    msg_size = task.get('message_sizes.inputs')
                    if msg_size is not None:
                        message_sizes.add(float(msg_size))
            
            # 获得节点中的最大资源
            max_cpu = max(node["cpu"] for node in node_resources.values())
            max_gpu = max(node["gpu"] for node in node_resources.values())
            
            # 生成所有可能的新配置组合
            new_records = []
            cpu_range = range(1, max_cpu + 1)
            gpu_range = range(0, max_gpu + 1)
            
            for msg_size in message_sizes:
                for cpu in cpu_range:
                    for gpu in gpu_range:
                        new_records.append((
                            cpu, gpu, msg_size, np.nan
                        ))
            
            # 将新记录转换为 numpy 结构化数组
            new_array = np.array(new_records, dtype=self.dtype)
            
            # 合并到现有数据库中
            if method in self.time_running_db:
                existing_array = self.time_running_db[method]
                # 使用 np.concatenate 合并
                combined_array = np.concatenate((existing_array, new_array))
                # 去重，确保唯一性
                self.time_running_db[method] = np.unique(combined_array, axis=0)
            else:
                # 如果当前方法尚无记录，直接赋值
                self.time_running_db[method] = new_array
            
            # 更新索引
            self._create_index(method)

                                
    def add_runtime_records(self, historical_data: Dict[str, List[Dict]]):
        """批量添加历史运行记录"""
        for method, tasks in historical_data.items():
            if not tasks:
                continue
                
            data = self.time_running_db[method]
            indices = self.method_indices[method]
            
            for task in tasks:
                cpu = task['resources.cpu']
                gpu = task['resources.gpu']
                # node = task['resources']['node']
                msg_size = task['message_sizes.inputs']
                
                # 使用索引快速定位记录
                key = (cpu, gpu)
                if key in indices:
                    mask = (data['message_sizes'] == msg_size) & \
                           (data['cpu'] == cpu) & \
                           (data['gpu'] == gpu)
                        #    (data['node'] == node)
                    if np.any(mask):
                        data['time_running'][mask] = task['time_running']
                        
    def fill_runtime_records_with_predictor(self, method=None):
        """通过预测模型补充数据库缺失值"""
        for method in self.methods:
            if method not in self._models:
                logger.info(f"predictor for method{method} not trained")
                continue
                
            data = self.time_running_db[method]
            
            # 找到所有缺失值的位置
            missing_mask = np.isnan(data['time_running'])
            missing_indices = np.where(missing_mask)[0]
            
            if len(missing_indices) == 0:
                logger.info(f"no NAN for method{method}")
                continue
                
            # 准备特征数据进行批量预测
            features = np.zeros((len(missing_indices), 3))
            features[:, 0] = data['message_sizes'][missing_indices]
            features[:, 1] = data['cpu'][missing_indices]
            features[:, 2] = data['gpu'][missing_indices]
            
            try:
                # 批量预测
                predictions = np.expm1(self._models[method].predict(features))
                # 填充预测值
                data['time_running'][missing_indices] = predictions
            except Exception as e:
                logger.error(f"Prediction failed for method {method}: {str(e)}")
                continue
    
    def visualize_runtime_trends(self, method: str, feature: str = 'message_sizes',
                            control_features: Dict[str, Any] = None,
                            show_scatter: bool = True):
        """
        使用matplotlib生成任务运行时间随特定特征变化的可视化图表
        
        Args:
            method: 要可视化的方法名
            feature: 要观察的特征名称 ('message_sizes', 'cpu', 'gpu')
            control_features: 其他特征的固定值，例如 {'cpu': 4, 'gpu': 1}
            show_scatter: 是否显示散点图
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn')
        
        data = self.time_running_db[method]
        if len(data) == 0:
            logger.warning(f"No data available for method {method}")
            return
            
        # 准备数据
        valid_mask = ~np.isnan(data['time_running'])
        valid_data = data[valid_mask]
        
        # 如果指定了控制特征，进一步过滤数据
        if control_features:
            for feat, value in control_features.items():
                valid_data = valid_data[valid_data[feat] == value]
                
        if len(valid_data) == 0:
            logger.warning("No valid data after filtering")
            return
                
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制散点图
        if show_scatter:
            plt.scatter(valid_data[feature], valid_data['time_running'], 
                    alpha=0.5, label='Actual Data')
        
        # 对特征值排序并计算平均运行时间趋势线
        unique_features = np.unique(valid_data[feature])
        avg_times = []
        for feat_value in unique_features:
            mask = valid_data[feature] == feat_value
            avg_time = np.mean(valid_data['time_running'][mask])
            avg_times.append(avg_time)
        
        # 绘制趋势线
        plt.plot(unique_features, avg_times, 'r-', linewidth=2, label='Average Trend')
        
        # 设置图表标题和标签
        title = f'Runtime Trend for {method} by {feature}'
        if control_features:
            title += f'\nControl Features: {control_features}'
        plt.title(title)
        plt.xlabel(feature)
        plt.ylabel('Runtime (s)')
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 优化布局
        plt.tight_layout()
        
        # 显示图表
        plt.show()

        
    @lru_cache(maxsize=10000) 
    def get_runtime(self, cpu, gpu, msg_size, method) -> Optional[float]:
        """获取运行时间"""
        if method not in self.methods:
            return None
            
        data = self.time_running_db[method]
        indices = self.method_indices[method]
        
        # 使用索引快速查找匹配的资源配置
        key = (cpu, gpu)
        if key not in indices:
            return None
            
        # 找到最接近的message_sizes值
        mask = (data['cpu'] == cpu) & \
                (data['gpu'] == gpu) 
        matches = data[mask]
        
        if len(matches) == 0:
            raise KeyError(f"get_runtime failed for features{(cpu, gpu, msg_size, method)}")
            
        # 找到最接近的message_sizes值对应的运行时间
        idx = np.abs(matches['message_sizes'] - msg_size).argmin()
        return float(matches[idx]['time_running'])

            
    def _init_model(self, method: str, model_type: Literal["random_forest", "polynomial"]):
        from sklearnex import patch_sklearn, unpatch_sklearn
        patch_sklearn()
        """为每个方法初始化模型pipeline"""
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=3)),
                ('linear_model', LinearRegression())
            ])
            
        self._models[method] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
    def train(self, train_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """统一的训练入口，返回每个方法的训练得分"""
        scores = {}
        for method, data in train_data.items():
            if len(data) < 5:  # 数据太少，跳过训练
                continue
                
            # 转换为numpy数组以提高性能
            X, y = self._prepare_training_data(data)
            if X is None or y is None:
                continue
                
            # 训练并记录得分
            try:
                self._models[method].fit(X, y)
                scores[method] = self._models[method].score(X, y)
            except Exception as e:
                logger.error(f"Training failed for method {method}: {str(e)}")
                continue
                
        return scores
    
    def _prepare_training_data(self, data: List[Dict]) -> tuple:
        """准备训练数据，返回(X, y)"""
        try:
            df = pd.DataFrame(data)
            df = df.dropna()
            
            # 提取特征和目标值
            X = df.drop(columns=['time_running', 'method', 'task_id'])
            y = np.log1p(df['time_running'])
            
            # 缓存特征统计信息供后续使用
            self._cache_feature_stats(df)
            
            return X.to_numpy(), y.to_numpy()
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return None, None
            
    def _cache_feature_stats(self, df: pd.DataFrame):
        """缓存特征统计信息"""
        for col in df.columns:
            if col not in ['time_running', 'method', 'task_id']:
                self._feature_stats[col] = (
                    float(df[col].mean()),
                    float(df[col].std())
                )
                
    @lru_cache(maxsize=1024)
    def _predict_single(self, method: str, feature_tuple: tuple) -> float:
        """预测单个任务的运行时间，使用tuple作为缓存key"""
        try:
            X = np.array(feature_tuple).reshape(1, -1)
            prediction = self._models[method].predict(X)[0]
            return np.expm1(prediction)
        except Exception as e:
            logger.error(f"Prediction failed for method {method}: {str(e)}")
            return None

    def estimate_time(self, task: Dict) -> Optional[float]:
        """预测单个任务的运行时间"""
        method = task['method']
        if method not in self._models:
            return None
            
        # 将特征转换为可哈希的tuple用于缓存
        features = self._extract_features(task)
        feature_tuple = tuple(features.values())
        
        return self._predict_single(method, feature_tuple)

    
    # def estimate_ga_population(self, population: List, sch_task_list: Dict, 
    #                         all_node: bool = False) -> None:
    #     """批量预测种群中所有任务的运行时间"""
        
    #     all_tasks = np.concatenate([ind.task_array for ind in population])
        
    #     method_to_tasks = {
    #         method: all_tasks[all_tasks['name'] == method] 
    #         for method in self.methods if method in self._models
    #     }
        
    #     for method, tasks in method_to_tasks.items():
    #         if len(tasks) == 0:
    #             continue
            
    #         try:
    #             num_tasks = len(tasks)
    #             # 一次性创建特征矩阵
    #             X = np.zeros((num_tasks, 3), dtype=np.float32)
                
    #             # 批量填充特征矩阵
    #             task_ids = tasks['task_id']
    #             X[:, 0] = [sch_task_list[tid].get('message_sizes.inputs', 1) for tid in task_ids]  # message sizes
    #             X[:, 1] = tasks['cpu']  # CPU
    #             X[:, 2] = tasks['gpu']  # GPU
                
    #             # 批量预测
    #             predictions = np.expm1(self._models[method].predict(X))
                
    #             # 创建任务ID到预测结果的映射
    #             prediction_map = dict(zip(task_ids, predictions))
                
    #             # 使用向量化操作更新种群
    #             for ind in population:
    #                 method_mask = ind.task_array['name'] == method
    #                 if np.any(method_mask):
    #                     task_ids = ind.task_array[method_mask]['task_id']
    #                     # 使用向量化操作更新运行时间
    #                     update_indices = [ind._task_id_index[tid] for tid in task_ids]
    #                     ind.task_array['total_runtime'][update_indices] = \
    #                         [prediction_map[tid] for tid in task_ids]
                        
    #         except Exception as e:
    #             logger.error(f"Method {method} batch预测失败: {str(e)}")
    #             logger.exception(e)
    
    def estimate_ga_population(self, population: List, sch_task_list: Dict, 
                         all_node: bool = False) -> None:
        """批量预测种群中所有任务的运行时间，优先使用数据库"""
        # 按方法分组处理任务
        method_tasks = defaultdict(list)
        for ind in population:
            for task in ind.task_array:
                method_tasks[task['name']].append((task, ind))
        
        # 批量处理每个方法的任务
        for method, tasks in method_tasks.items():
            if method not in self.methods:
                continue
                
            # 收集需要模型预测的任务
            model_prediction_tasks = []
            
            for task, ind in tasks:
                cpu = task['cpu']
                gpu = task['gpu']
                # 'node': task['node']
                
                msg_size = sch_task_list[task['task_id']].get(
                    'message_sizes.inputs', 0)
                    
                runtime = self.get_runtime(cpu, gpu, msg_size, method)
                
                if runtime is not None:
                    idx = ind._task_id_index[task['task_id']]
                    ind.task_array[idx]['total_runtime'] = runtime
                
    def _extract_features(self, task: Dict) -> Dict:
        """提取任务特征"""
        features = {}
        for feature in self.features['default']:
            if feature not in ['time_running', 'method', 'task_id']:
                value = self._get_nested_value(task, feature)
                features[feature] = value
        return features
    
    @staticmethod
    def _get_nested_value(dict_obj: Dict, key_path: str) -> Any:
        """获取嵌套字典中的值"""
        current = dict_obj
        for key in key_path.split('.'):
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current
    
    # def _extract_features_from_sch_task(self, sch_task: Dict, 
    #                                   resources: Dict) -> Dict:
    #     """从调度任务中提取特征"""
    #     features = sch_task.copy()
    #     # 更新资源相关的特征
    #     features['resources.cpu'] = resources['cpu']
    #     features['resources.gpu'] = resources['gpu']
    #     return features

    
    # def save_models(self, path: str):
    #     """保存训练好的模型"""
    #     import joblib
    #     save_dict = {
    #         'models': self._models,
    #         'feature_stats': self._feature_stats,
    #         'model_type': self.model_type
    #     }
    #     joblib.dump(save_dict, path)
    
    # @classmethod
    # def load_models(cls, path: str, methods: List[str], 
    #                features: Dict[str, List[str]]):
    #     """加载保存的模型"""
    #     import joblib
    #     saved_dict = joblib.load(path)
        
    #     predictor = cls(
    #         methods=methods,
    #         features=features,
    #         model_type=saved_dict['model_type']
    #     )
    #     predictor._models = saved_dict['models']
    #     predictor._feature_stats = saved_dict['feature_stats']
    #     return predictor



class FCFSScheduler:
    def __init__(
        self,
        resources: dict = None,
        at: available_task = None,
        hist_data: HistoricalData = None,
        sch_data: Sch_data = None
    ):
        self.node_resources = resources
        self.resources = copy.deepcopy(self.node_resources)
        for key, value in self.resources.items():
            value['gpu_devices'] = list(range(value['gpu']))
        self.at = at
        self.sch_data = sch_data
        self.running_task_node = defaultdict(list)
        self.current_time = 0
        
        # priority queue for task
        self.task_queue = []
        
        self.run_lock = threading.Lock()

    def generate_node(self):
        """Round-robin node selection"""
        nodes = list(self.node_resources.keys())
        index = 0
        while True:
            yield nodes[index]
            index = (index + 1) % len(nodes)

    def allocate_basic_resources(self, task_name, task_id):
        """Allocate basic resources based on task requirements"""
        predefine_cpu = self.sch_data.sch_task_list[task_id]['resources.cpu']
        predefine_gpu = self.sch_data.sch_task_list[task_id]['resources.gpu']
        return {
            "cpu": predefine_cpu,
            "gpu": predefine_gpu
        }

    def find_available_node(self, required_resources):
        """Find first available node with enough resources"""
        for node, resources in self.resources.items():
            if (resources['cpu'] >= required_resources['cpu'] and 
                resources['gpu'] >= required_resources['gpu']):
                return node
        return None
    
    # allocate and recover resources, with threadlock
    def allocate_resources(self, result_obj:Result):
        with self.run_lock:
            cpu_value = getattr(result_obj.resources, 'cpu')
            gpu_value = getattr(result_obj.resources, 'gpu')
            node = getattr(result_obj.resources, 'node')
            task_id = result_obj.task_id
            name = result_obj.method
            self.resources[node]['cpu'] -= cpu_value
            self.resources[node]['gpu'] -= gpu_value
            gpu_value, self.resources[node]['gpu_devices'] = (
                self.resources[node]['gpu_devices'][
                    :gpu_value
                ],
                self.resources[node]['gpu_devices'][
                    gpu_value:
                ],
            )
            
            result_obj.inputs[1]['gpu'] = gpu_value
            result_obj.inputs[1]['cpu'] = cpu_value
    
    def recover_resources(self, result_obj:Result):
        with self.run_lock:
            node = getattr(result_obj.resources, 'node')
            gpu_value = result_obj.inputs[1]['gpu']
            self.resources[node]['cpu'] += result_obj.resources.cpu
            self.resources[node]['gpu'] += len(gpu_value)
            self.resources[node]['gpu_devices'].extend(gpu_value)
            # for task in  self.running_task_node[node]:
            #     if task['task_id'] == result_obj.task_id:
            #         self.running_task_node[node].remove(task)

    def run_fcfs(self, all_tasks: list[dict[str, int]]) -> list:
        """
        FCFS调度算法的主要实现
        按照任务到达顺序分配资源和节点
        """
        logger.info(f"Starting FCFS with available tasks: {all_tasks}")
        
        task_allocation = []
        which_node = self.generate_node()
        
        # 按照任务到达顺序处理
        for task_name, task_ids in all_tasks.items():
            for task_id in task_ids:
                # 获取基本资源需求
                basic_resources = self.allocate_basic_resources(task_name, task_id)
                
                # 寻找可用节点
                node = self.find_available_node(basic_resources)
                if node is None:
                    logger.warning(f"No available resources for task {task_name}:{task_id}")
                    continue
                
                # 创建任务分配
                new_task = {
                    "name": task_name,
                    "task_id": task_id,
                    "resources": {
                        "cpu": basic_resources['cpu'],
                        "gpu": basic_resources['gpu'],
                        "node": node
                    }
                }
                
                # 更新资源使用情况
                self.resources[node]['cpu'] -= basic_resources['cpu']
                self.resources[node]['gpu'] -= basic_resources['gpu']
                
                # 添加到分配列表
                task_allocation.append(new_task)
                
                logger.info(f"Allocated task {task_name}:{task_id} to node {node}")
                
        return task_allocation

# class Back_Filing_Scheduler:
#     pass


@jit(nopython=True)
def _calculate_completion_time(
    task_cpu,        # shape: (n,), dtype: int32
    task_gpu,        # shape: (n,), dtype: int32
    task_runtime,    # shape: (n,), dtype: float64
    running_finish_times,  # shape: (m,), dtype: float64
    running_cpus,         # shape: (m,), dtype: int32
    running_gpus,         # shape: (m,), dtype: int32
    resources_cpu,   # int
    resources_gpu,   # int
    current_time,    # float
):
    """Numba优化版本的完成时间计算"""
    # 预分配数组
    n_tasks = len(task_cpu)
    n_running = len(running_finish_times)
    max_tasks = n_tasks + n_running
    
    ongoing_times = np.zeros(max_tasks, dtype=np.float64)
    ongoing_cpus = np.zeros(max_tasks, dtype=np.int32)
    ongoing_gpus = np.zeros(max_tasks, dtype=np.int32)
    
    # 初始化资源
    avail_cpu = resources_cpu
    avail_gpu = resources_gpu
    task_count = 0
    start_time = current_time
    
    # 添加运行中任务
    for i in range(n_running):
        insert_pos = task_count
        for j in range(task_count):
            if ongoing_times[j] > running_finish_times[i]:
                insert_pos = j
                break
        # 移动现有任务
        for j in range(task_count, insert_pos, -1):
            ongoing_times[j] = ongoing_times[j - 1]
            ongoing_cpus[j] = ongoing_cpus[j - 1]
            ongoing_gpus[j] = ongoing_gpus[j - 1]
        
        ongoing_times[insert_pos] = running_finish_times[i]
        ongoing_cpus[insert_pos] = running_cpus[i]
        ongoing_gpus[insert_pos] = running_gpus[i]
        avail_cpu -= ongoing_cpus[insert_pos]
        avail_gpu -= ongoing_gpus[insert_pos]
        task_count += 1
        
    # 记录资源使用变化点
    changes_times = np.zeros(max_tasks * 2, dtype=np.float64)
    changes_cpu = np.zeros(max_tasks * 2, dtype=np.int32)
    changes_gpu = np.zeros(max_tasks * 2, dtype=np.int32)
    changes_count = 0
    
    if task_count > 0:
        changes_times[0] = current_time
        changes_cpu[0] = resources_cpu - avail_cpu
        changes_gpu[0] = resources_gpu - avail_gpu
        changes_count += 1

    # 处理任务数组
    task_starts = np.zeros(n_tasks, dtype=np.float64)
    task_ends = np.zeros(n_tasks, dtype=np.float64)
    
    for i in range(n_tasks):
        required_cpu = task_cpu[i]
        required_gpu = task_gpu[i]
        duration = task_runtime[i]
        
        # 检查已完成任务
        while task_count > 0 and ongoing_times[0] <= current_time:
            avail_cpu += ongoing_cpus[0]
            avail_gpu += ongoing_gpus[0]
            
            # 记录资源变化
            changes_times[changes_count] = current_time 
            changes_cpu[changes_count] = resources_cpu - avail_cpu
            changes_gpu[changes_count] = resources_gpu - avail_gpu
            changes_count += 1
            
            # 移除完成的任务
            for j in range(task_count - 1):
                ongoing_times[j] = ongoing_times[j + 1]
                ongoing_cpus[j] = ongoing_cpus[j + 1]
                ongoing_gpus[j] = ongoing_gpus[j + 1]
            task_count -= 1
            
        # 等待资源
        while avail_cpu < required_cpu or avail_gpu < required_gpu:
            if task_count == 0:
                return -1.0, -1.0, -1.0, task_starts, task_ends
            current_time = ongoing_times[0]
            avail_cpu += ongoing_cpus[0]
            avail_gpu += ongoing_gpus[0]
            
            # 记录资源变化
            changes_times[changes_count] = current_time
            changes_cpu[changes_count] = resources_cpu - avail_cpu
            changes_gpu[changes_count] = resources_gpu - avail_gpu
            changes_count += 1
            
            # 移除第一个任务
            for j in range(task_count - 1):
                ongoing_times[j] = ongoing_times[j + 1]
                ongoing_cpus[j] = ongoing_cpus[j + 1]
                ongoing_gpus[j] = ongoing_gpus[j + 1]
            task_count -= 1
            
        # 分配新任务
        finish_time = current_time + duration
        
        # 二分查找插入位置
        insert_pos = task_count
        for j in range(task_count):
            if ongoing_times[j] > finish_time:
                insert_pos = j
                break
                
        # 移动现有任务
        for j in range(task_count, insert_pos, -1):
            ongoing_times[j] = ongoing_times[j - 1]
            ongoing_cpus[j] = ongoing_cpus[j - 1]
            ongoing_gpus[j] = ongoing_gpus[j - 1]
            
        # 插入新任务
        ongoing_times[insert_pos] = finish_time
        ongoing_cpus[insert_pos] = required_cpu
        ongoing_gpus[insert_pos] = required_gpu
        task_count += 1
        
        # 记录任务时间
        task_starts[i] = current_time
        task_ends[i] = finish_time
        
        avail_cpu -= required_cpu
        avail_gpu -= required_gpu
        
        # 记录资源变化
        changes_times[changes_count] = current_time
        changes_cpu[changes_count] = resources_cpu - avail_cpu
        changes_gpu[changes_count] = resources_gpu - avail_gpu
        changes_count += 1
    
    while task_count > 0:
        current_time = ongoing_times[0]
            
        avail_cpu += ongoing_cpus[0]
        avail_gpu += ongoing_gpus[0]
        
        # 记录资源变化
        changes_times[changes_count] = current_time
        changes_cpu[changes_count] = resources_cpu - avail_cpu
        changes_gpu[changes_count] = resources_gpu - avail_gpu
        changes_count += 1
        
        # 移除完成的任务
        for j in range(task_count - 1):
            ongoing_times[j] = ongoing_times[j + 1]
            ongoing_cpus[j] = ongoing_cpus[j + 1]
            ongoing_gpus[j] = ongoing_gpus[j + 1]
        task_count -= 1
        
    # 计算资源使用面积
    resource_area = 0.0
    for i in range(changes_count - 1):
        time_delta = changes_times[i + 1] - changes_times[i]
        area = (changes_cpu[i] + changes_gpu[i]) * time_delta
        resource_area += area
        
    completion_time = current_time - start_time if n_tasks > 0 else 0
    total_runtime = completion_time
    
    return completion_time, resource_area, total_runtime, task_starts, task_ends


# multiprocessing, class should be pickleable
class evosch2:
    """add all task in individual
    cost function calculate total time idle time
    """

    def __init__(
        self,
        resources: dict = None,
        at: available_task = None,
        hist_data:HistoricalData = None,
        sch_data: Sch_data = None,
        population_size=10,
    ):
        # self.his_population = set() # should following round consider history evo result?
        self.node_resources: dict = resources # total resources for all node
        self.resources: dict = copy.deepcopy(
            self.node_resources
        )  # used in colmena base.py.  data: {"node1":{"cpu":56,"gpu":4},"node2":{"cpu":56,"gpu":4}} gpu is ids like [0, 1, 2, 3]
        for key, value in self.resources.items():
            value['gpu_devices'] = list(
                range(value['gpu'])
            )  # TODO gpu nums change to gpu_devices ids; next we need get ids from config
        self.resources_evo: dict = copy.deepcopy(self.resources)  # used in evosch
        logger.info("total resources: {}".format(self.resources_evo))
        self.sch_data: Sch_data = sch_data
        self.at: available_task = at  # available task
        self.population = []  # [individual,,,] # store all individual on single node
        self.population_node = defaultdict(
            list
        )  # {node: [individual,,,],,,} # store all individual on all node

        ## log the running task for track the resource and time
        self.running_task: list[dict[str, int]] = (
            []
        )  # {'task_id': 1, 'name': 'simulate', 'start_time': 100, 'finish_time': 200, 'total_time': 100, resources:{'cpu':3,'gpu':0}}
        self.running_task_node: dict = defaultdict(list)
        self.current_time = (
            0  # current running time for compute  while trigger evo_scheduler
        )
        self.best_ind: individual = None

        # 添加日志相关的初始化
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.usr_path = os.path.expanduser("~")
        self.log_dir = f"{self.usr_path}/project/colmena/multisite_/finetuning-surrogates/job_out"
        self.log_path = os.path.join(self.log_dir, f'run_ga_{self.timestamp}.log')

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化日志文件
        with open(self.log_path, 'w') as f:
            f.write(f"GA Optimization Started at {self.timestamp}\n")
            f.write("=" * 80 + "\n")

    def write_log(self, message):
        with open(self.log_path, 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")

    # allocate and recover resources, with threadlock
    def allocate_resources(self, result_obj:Result):
        with self.run_lock:
            cpu_value = result_obj['resources']['cpu']
            gpu_value = result_obj['resources']['gpu']
            self.resources['node']['cpu'] -= cpu_value
            self.resources['node']['gpu'] -= gpu_value
            #TODO self.at.remove_task_id(task_name=result_obj['name'], task_id=result_obj['task_id']) 判断是移除可用任务还是已经调度任务

    def recover_resources(self, result_obj:Result):
        with self.run_lock:
            node = getattr(result_obj.resources, 'node')
            gpu_value = result_obj.inputs[1]['gpu']
            self.resources['node']['cpu'] += result_obj.resources.cpu
            self.resources['node']['gpu'] += len(gpu_value)
            self.resources['node']['gpu_devices'].extend(gpu_value)
            for task in  self.running_task_node[node]:
                if task['task_id'] == result_obj.task_id:
                    self.running_task_node[node].remove(task)

    def get_dict_list_nums(self, dict_list: dict):
        """
        get total task nums from task allocation
        """
        task_nums = 0
        for key, value in dict_list.items():
            task_nums += len(value)
        return task_nums

    def get_resources(self):
        return self.resources

    def get_total_resources(self):
        aggregated_resources = {}
        # gateher total resources from all node
        for resource in self.node_resources:
            for key, value in self.node_resources[resource].items():
                if key in aggregated_resources:
                    aggregated_resources[key] += value
                else:
                    aggregated_resources[key] = value
        return aggregated_resources


    def detect_no_his_task(self, all_tasks, total_nums=5):
        '''检测没有历史数据的任务并生成初始运行样本'''
        if all_tasks is None:
            return None
            
        # 计算需要采样的任务总数
        total_sample_tasks = 0
        for name, ids in all_tasks.items():
            avail = len(ids)
            if avail == 0:
                continue
                
            hist = len(self.sch_data.historical_task_data.historical_data[name])
            if hist < total_nums:
                sample_nums = min(total_nums - hist, avail)
                total_sample_tasks += sample_nums
        
        if total_sample_tasks == 0:
            return None
            
        # 创建individual实例
        ind = individual(tasks_nums=total_sample_tasks, total_resources=self.node_resources)
        which_node = self.generate_node()
        task_idx = 0
        
        for name, ids in all_tasks.items():
            avail = len(ids)
            if avail == 0:
                continue
                
            hist = len(self.sch_data.historical_task_data.historical_data[name])
            if hist < total_nums:
                # 确定需要采样的数量
                sample_nums = min(total_nums - hist, avail)
                
                # 获取预定义资源
                predefine_cpu = self.sch_data.sch_task_list[ids[0]]['resources.cpu']
                predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
                
                # CPU采样范围
                cpu_lower_bound = min(2, predefine_cpu // 2)
                cpu_upper_bound = max(res['cpu'] for res in self.node_resources.values())
                cpu_choices = np.linspace(
                    cpu_lower_bound,
                    cpu_upper_bound,
                    num=sample_nums,
                    endpoint=True,
                    dtype=int
                )
                
                # GPU采样范围
                if predefine_gpu == 0:
                    gpu_choices = np.zeros(sample_nums, dtype=int)
                else:
                    gpu_lower_bound = 1
                    gpu_upper_bound = max(res['gpu'] for res in self.node_resources.values())
                    gpu_choices = np.linspace(
                        gpu_lower_bound,
                        gpu_upper_bound,
                        num=sample_nums,
                        endpoint=True,
                        dtype=int
                    )
                
                # 填充task_array
                for i in range(sample_nums):
                    node = next(which_node)
                    ind.task_array[task_idx] = (
                        name,           # name
                        ids[i],         # task_id
                        cpu_choices[i], # cpu
                        gpu_choices[i], # gpu
                        node,           # node
                        1.0,           # total_runtime (默认值)
                        0.0,           # start_time
                        0.0            # finish_time
                    )
                    task_idx += 1
        
        # 更新索引和node_array
        ind.update_task_id_index()
        
        # 初始化node_array
        # nodes = set(ind.task_array['node'])
        # ind.node_array = np.zeros(len(nodes), dtype=ind.node_dtype)
        
        # for i, node in enumerate(nodes):
        #     ind.node_array[i]['node'] = node
        #     node_mask = ind.task_array['node'] == node
        #     task_indices = np.where(node_mask)[0]
        #     ind.node_array[i]['task_indices'][:len(task_indices)] = task_indices
        #     ind.node_array[i]['task_indices'][len(task_indices):] = -1
        
        return ind

    def calc_time(self, tasks: np.ndarray) -> tuple[float, float]:
        """计算任务的CPU和GPU总时间
        
        Args:
            tasks: 任务数组
        
        Returns:
            tuple: (CPU总时间, GPU总时间)
        """
        total_cpu_time = np.sum(tasks['cpu'] * tasks['total_runtime'])
        total_gpu_time = np.sum(tasks['gpu'] * tasks['total_runtime'])
        return total_cpu_time, total_gpu_time

    def calc_utilization(self, ind: individual) -> tuple[dict, dict, dict]:
        """计算各节点的资源利用情况
        
        Args:
            ind: individual对象
        
        Returns:
            tuple: (CPU时间字典, GPU时间字典, 完成时间字典)
        """
        total_cpu_time = defaultdict(float)
        total_gpu_time = defaultdict(float)
        completion_time = defaultdict(float)
        
        for node in self.node_resources.keys():
            node_mask = ind.task_array['node'] == node
            node_tasks = ind.task_array[node_mask]
            
            total_cpu_time[node], total_gpu_time[node] = self.calc_time(node_tasks)
            
            completion_time[node], _, _ = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
                self.running_task_node[node],
                node_tasks,
                ind
            )

        return total_cpu_time, total_gpu_time, completion_time

    def load_balance(self, ind: individual) -> None:
        """平衡各节点的负载
        
        Args:
            ind: individual对象
        """
        if not isinstance(ind, individual):
            raise ValueError("load_balance input is not individual")

        total_cpu_time, total_gpu_time, completion_time = self.calc_utilization(ind)
        diff_cur = max(completion_time.values()) - min(completion_time.values())
        diff_pre = max(completion_time.values())
        max_pre = max(completion_time.values())
        max_node = max(completion_time, key=completion_time.get)
        min_node = min(completion_time, key=completion_time.get)

        while diff_cur < diff_pre:
            max_node = max(completion_time, key=completion_time.get)
            min_node = min(completion_time, key=completion_time.get)

            best_task_idx = None
            best_diff = float('inf')

            # 获取最大负载节点的任务
            max_node_mask = ind.task_array['node'] == max_node
            max_node_tasks = ind.task_array[max_node_mask]
            
            # 遍历最大负载节点的每个任务
            for i, task in enumerate(max_node_tasks):
                # 计算移动任务后的资源使用情况
                used_cpu_time_max = total_cpu_time[max_node] - task['cpu'] * task['total_runtime']
                total_cpu_area_max = completion_time[max_node] * self.resources_evo[max_node]['cpu']

                temp_gpu_time_max = total_gpu_time[max_node] - task['gpu'] * task['total_runtime']
                total_gpu_area_max = completion_time[max_node] * self.resources_evo[max_node]['gpu']

                temp_cpu_time_min = total_cpu_time[min_node] + task['cpu'] * task['total_runtime']
                total_cpu_area_min = completion_time[min_node] * self.resources_evo[min_node]['cpu']

                temp_gpu_time_min = total_gpu_time[min_node] + task['gpu'] * task['total_runtime']
                total_gpu_area_min = completion_time[min_node] * self.resources_evo[min_node]['gpu']

                temp_max_diff = abs(total_cpu_area_max - used_cpu_time_max) + abs(total_gpu_area_max - temp_gpu_time_max)
                temp_min_diff = abs(total_cpu_area_min - temp_cpu_time_min) + abs(total_gpu_area_min - temp_gpu_time_min)
                combined_diff = temp_max_diff + temp_min_diff

                if combined_diff < best_diff:
                    best_diff = combined_diff
                    best_task_idx = i

            if best_task_idx is not None:
                # 更新任务节点分配
                task_to_move = max_node_tasks[best_task_idx]
                
                # 更新task_array中的节点信息
                task_id = task_to_move['task_id']
                ind_idx = ind.get_task_index(task_id)
                ind.task_array[ind_idx]['node'] = min_node
                
                # 更新资源使用时间
                total_cpu_time[max_node] -= task_to_move['cpu'] * task_to_move['total_runtime']
                total_gpu_time[max_node] -= task_to_move['gpu'] * task_to_move['total_runtime']
                total_cpu_time[min_node] += task_to_move['cpu'] * task_to_move['total_runtime']
                total_gpu_time[min_node] += task_to_move['gpu'] * task_to_move['total_runtime']

                # 重新计算完成时间
                max_node_mask = ind.task_array['node'] == max_node
                min_node_mask = ind.task_array['node'] == min_node
                
                completion_time[max_node], _, _ = self.calculate_completion_time_record_with_running_task(
                    self.node_resources[max_node],
                    self.running_task_node[max_node],
                    ind.task_array[max_node_mask],
                    ind
                )
                
                completion_time[min_node], _, _ = self.calculate_completion_time_record_with_running_task(
                    self.node_resources[min_node],
                    self.running_task_node[min_node],
                    ind.task_array[min_node_mask],
                    ind
                )

                diff_pre = diff_cur
                max_now = max(completion_time.values())
                diff_cur = max(completion_time.values()) - min(completion_time.values())
                
                if max_now > max_pre:
                    break
                else:
                    max_pre = max_now
            else:
                break
            
        ind.init_node_array()


    def calculate_completion_time_record_with_running_task(self, resources, running_tasks, task_array, ind):
        """计算完成时间并更新任务时间记录"""
        
        # 提取简单数组
        task_cpu = np.array([task['cpu'] for task in task_array], dtype=np.int32)
        task_gpu = np.array([task['gpu'] for task in task_array], dtype=np.int32)
        task_runtime = np.array([task['total_runtime'] for task in task_array], dtype=np.float64)
        
        # 转换running_tasks为简单数组
        if running_tasks:
            running_finish_times = np.array([task['finish_time'] for task in running_tasks], dtype=np.float64)
            running_cpus = np.array([task['resources']['cpu'] for task in running_tasks], dtype=np.int32)
            running_gpus = np.array([task['resources']['gpu'] for task in running_tasks], dtype=np.int32)
        else:
            running_finish_times = np.array([], dtype=np.float64)
            running_cpus = np.array([], dtype=np.int32)
            running_gpus = np.array([], dtype=np.int32)
        
        # 调用numba优化函数
        completion_time, resource_area, total_runtime, starts, ends = _calculate_completion_time(
            task_cpu,
            task_gpu,
            task_runtime,
            running_finish_times,
            running_cpus,
            running_gpus,
            resources['cpu'],
            resources['gpu'],
            time.time()
        )
        
        if completion_time < 0:
            raise ValueError("Resource allocation failed")
            
        # 更新individual中的任务时间
        for i, task in enumerate(task_array):
            idx = ind._task_id_index[task['task_id']]
            ind.task_array[idx]['start_time'] = starts[i]
            ind.task_array[idx]['finish_time'] = ends[i]
            
        return completion_time, resource_area, total_runtime

    def calculate_total_time(self, ind: individual):
        total_time = 0
        for task in ind.task_array:
            # total_time += self.hist_data.estimate_time(task)
            total_time += task['total_runtime']
        return total_time

    def fitness(self, ind: individual) -> float:
        """计算个体适应度
        
        Args:
            ind: individual对象
        
        Returns:
            float: 适应度分数
        """
        # unique_nodes = np.unique(ind.task_array['node'])
        unique_nodes = self.node_resources.keys()
        completion_times = np.zeros(len(unique_nodes))
        resource_areas = np.zeros(len(unique_nodes))
        total_runtimes = np.zeros(len(unique_nodes))
        
        for i, node in enumerate(unique_nodes):
            node_mask = ind.task_array['node'] == node
            node_tasks = ind.task_array[node_mask]
            
            completion_time, resource_area, total_runtime = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
                self.running_task_node[node],
                node_tasks,
                ind  # 传入individual对象
            )
            
            completion_times[i] = completion_time
            resource_areas[i] = resource_area
            total_runtimes[i] = total_runtime
        
        # 存储调度指标
        ind.completion_time = np.max(completion_times)
        ind.resource_area = np.sum(resource_areas)
        ind.total_runtime = np.max(total_runtimes)
        
        # 计算适应度分数
        ind.score = -ind.completion_time
        return ind.score

    def generate_node(self):
        nodes: list = list(self.node_resources.keys())
        index = 0
        while True:
            yield nodes[index]
            index = (index + 1) % len(nodes)

    def generate_population_all(self, all_tasks, population_size: int):
        def find_suitable_node(required_cpu, required_gpu, node_iterator):
            """寻找满足资源需求的节点，并将任务均匀分配"""
            checked_nodes = set()
            while True:
                try:
                    node = next(node_iterator)
                    if node in checked_nodes:  # 所有节点都检查过了
                        raise ValueError(f"No node available for task requiring CPU:{required_cpu}, GPU:{required_gpu}")
                    
                    checked_nodes.add(node)
                    if (self.node_resources[node]['cpu'] >= required_cpu and 
                        self.node_resources[node]['gpu'] >= required_gpu):
                        return node
                except StopIteration:  # 迭代器用完后重新开始
                    node_iterator = self.generate_node()

        which_node = self.generate_node()
        task_nums = sum(len(ids) for ids in all_tasks.values())
        
        cpu_upper_bound = min(16, max(res['cpu'] for res in self.node_resources.values()))
        gpu_upper_bound = min(4, max(res['gpu'] for res in self.node_resources.values()))
        
        population = []

        # 生成population_size个随机资源分配的个体
        for _ in range(population_size):
            ind = individual(tasks_nums=task_nums, total_resources=self.node_resources)
            task_idx = 0
            which_node = self.generate_node()
            
            # 直接构建task_array
            for name, ids in all_tasks.items():
                if not ids:
                    continue
                    
                predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
                for task_id in ids:
                    required_cpu = random.randint(1, cpu_upper_bound)
                    required_gpu = random.randint(1, gpu_upper_bound) if predefine_gpu > 0 else 0
                    node = find_suitable_node(required_cpu, required_gpu, which_node)
                    
                    ind.task_array[task_idx] = (
                        name, task_id, required_cpu, required_gpu, node, 0.0, 0.0, 0.0
                    )
                    task_idx += 1
            
            ind.task_array_shuffled()  # shuffle task array
            ind.update_task_id_index()
            ind.init_node_array()
            population.append(ind)

        # 添加最小资源配置的个体
        ind = individual(tasks_nums=task_nums, total_resources=self.node_resources)
        task_idx = 0
        which_node = self.generate_node()
        
        for name, ids in all_tasks.items():
            if not ids:
                continue
                
            predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
            for task_id in ids:
                required_cpu = 1
                required_gpu = min(1, predefine_gpu)
                node = find_suitable_node(required_cpu, required_gpu, which_node)
                
                ind.task_array[task_idx] = (
                    name, task_id, required_cpu, required_gpu, node, 0.0, 0.0, 0.0
                )
                task_idx += 1
        
        ind.task_array_shuffled()  # shuffle task array
        ind.update_task_id_index()
        ind.init_node_array()
        population.append(ind)

        # 添加预定义资源配置的个体
        ind = individual(tasks_nums=task_nums, total_resources=self.node_resources)
        task_idx = 0
        which_node = self.generate_node()
        
        for name, ids in all_tasks.items():
            if not ids:
                continue
                
            predefine_cpu = self.sch_data.sch_task_list[ids[0]]['resources.cpu']
            predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
            
            for task_id in ids:
                node = find_suitable_node(predefine_cpu, predefine_gpu, which_node)
                ind.task_array[task_idx] = (
                    name, task_id, predefine_cpu, predefine_gpu, node, 0.0, 0.0, 0.0
                )
                task_idx += 1
        
        ind.task_array_shuffled()  # shuffle task array
        ind.update_task_id_index()
        ind.init_node_array()
        population.append(ind)

        return population

    def generate_population_in_node(self, ind: individual, pop_size: int = 10):
        """为每个节点生成子种群"""
        population_node = defaultdict(list)
        ind.init_node_array()
        # 获取每个节点的任务数量
        for node_entry in ind.node_array:
            node = node_entry['node']
            valid_indices = node_entry['task_indices'][node_entry['task_indices'] >= 0]
            task_nums = len(valid_indices)
            
            if task_nums == 0:
                continue
                
            # 获取该节点的任务
            node_tasks = ind.task_array[valid_indices]
            
            # 为该节点生成多个个体
            for _ in range(pop_size):
                n_ind = individual(
                    tasks_nums=task_nums,
                    total_resources=self.node_resources[node],
                )
                
                # 复制任务数据
                n_ind.task_array = np.copy(node_tasks)
                np.random.shuffle(n_ind.task_array)  # 随机打乱顺序
                n_ind.update_task_id_index()
                
                population_node[node].append(n_ind)

            # 计算适应度
            scores = [self.fitness(ind) for ind in population_node[node]]

        return population_node
    
    def mutate_resources(self, population: list, ind: individual):
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        task_idx = random.choice(range(len(task_array)))
        cpu_choice = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
        cpu_delta = random.choice(cpu_choice)
        
        if task_array[task_idx]['gpu'] == 0:
            # task does not use GPU, skip
            pass
        else:
            gpu_choice = [-1, 0, 1]
            gpu_delta = random.choice(gpu_choice)
            new_gpu_value = task_array[task_idx]['gpu'] + gpu_delta
            new_gpu_value = max(1, new_gpu_value)
            new_gpu_value = min(new_gpu_value, self.node_resources[task_array[task_idx]['node']]['gpu'])
            task_array[task_idx]['gpu'] = new_gpu_value
        
        new_cpu_value = task_array[task_idx]['cpu'] + cpu_delta
        new_cpu_value = max(1, new_cpu_value)
        new_cpu_value = min(new_cpu_value, self.node_resources[task_array[task_idx]['node']]['cpu'])
        task_array[task_idx]['cpu'] = new_cpu_value
        
                # 验证没有重复task_id
        task_ids = [task['task_id'] for task in new_ind.task_array]
        assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after copy: {task_ids}"
        population.append(new_ind)

    
    def mutate_seq(self, population: list, ind: individual):
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        # 使用正确的numpy结构化数组交换方式
        idx1, idx2 = np.random.choice(range(len(task_array)), size=2, replace=False)
        temp = task_array[idx1].copy()  # 使用深拷贝
        task_array[idx1] = task_array[idx2].copy()  # 使用深拷贝
        task_array[idx2] = temp
        
        new_ind.update_task_id_index()
        
        # 验证没有重复task_id
        task_ids = [task['task_id'] for task in task_array]
        assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after mutate_seq: {task_ids}"
        
        population.append(new_ind)
    
    def crossover_arith_ave(self, population: list, ind1: individual, ind2: individual):
        """算术平均交叉 - 直接在task_array上操作"""
        new_ind = ind1.copy()
        try:
            for task_id in ind1._task_id_index:
                idx1 = ind1._task_id_index[task_id]
                idx2 = ind2._task_id_index[task_id]
                new_ind.task_array[idx1]['cpu'] = (
                    ind1.task_array[idx1]['cpu'] + ind2.task_array[idx2]['cpu']
                ) // 2
                new_ind.task_array[idx1]['gpu'] = (
                    ind1.task_array[idx1]['gpu'] + ind2.task_array[idx2]['gpu']
                ) // 2
        except KeyError:
            print(f"Task {task_id} not found in both individuals")
            print(ind1.task_array)
            print(ind2.task_array)
            return
        
                # 验证没有重复task_id
        task_ids = [task['task_id'] for task in new_ind.task_array]
        assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after copy: {task_ids}"
        population.append(new_ind)
    
    def crossover_pmx(self, population: list, ind1: individual, ind2: individual):
        """PMX交叉 - 使用numpy数组操作"""
        size = len(ind1.task_array)
        if size < 2:
            return
        
        new_ind1 = ind1.copy()
        new_ind2 = ind2.copy()
        
        # 选择交叉点
        cxpoint1, cxpoint2 = sorted(np.random.choice(size, 2, replace=False))
        
        # 创建交换区域的映射
        # 使用task_id作为键创建映射关系
        segment1 = ind1.task_array[cxpoint1:cxpoint2+1]
        segment2 = ind2.task_array[cxpoint1:cxpoint2+1]
        
        # 存储交换段中task_id的对应关系
        mapping1 = {t['task_id']: i for i, t in enumerate(segment1, cxpoint1)}
        mapping2 = {t['task_id']: i for i, t in enumerate(segment2, cxpoint1)}
        
        # 交换中间段
        temp_segment = segment1.copy()
        new_ind1.task_array[cxpoint1:cxpoint2+1] = segment2
        new_ind2.task_array[cxpoint1:cxpoint2+1] = temp_segment
        
        # 处理交叉段外的元素
        for i in range(size):
            if i < cxpoint1 or i > cxpoint2:
                # 处理ind1
                current_task_id = new_ind1.task_array[i]['task_id']
                cnt = 0
                while current_task_id in mapping2:
                    if cnt > size:
                        raise ValueError("Infinite loop detected,ind1 {} ind2 {}, current_task_id".format(ind1, ind2, current_task_id))
                    cnt += 1
                    # 找到映射关系中对应的位置
                    mapped_idx = mapping2[current_task_id]
                    current_task_id = ind1.task_array[mapped_idx]['task_id']
                # 找到最终的任务后，复制所有字段
                new_ind1.task_array[i] = ind1.task_array[
                    ind1._task_id_index[current_task_id]
                ].copy()
                
                # 处理ind2
                current_task_id = new_ind2.task_array[i]['task_id']
                cnt = 0
                while current_task_id in mapping1:
                    if cnt > size:
                        raise ValueError("Infinite loop detected,ind1 {} ind2 {}, current_task_id".format(ind1, ind2, current_task_id))
                    cnt += 1
                    mapped_idx = mapping1[current_task_id]
                    current_task_id = ind2.task_array[mapped_idx]['task_id']
                new_ind2.task_array[i] = ind2.task_array[
                    ind2._task_id_index[current_task_id]
                ].copy()
        
        # 更新索引
        new_ind1.update_task_id_index()
        new_ind2.update_task_id_index()
        
                # 验证没有重复task_id
        task_ids = [task['task_id'] for task in new_ind1.task_array]
        assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after copy: {task_ids}"
        
                # 验证没有重复task_id
        task_ids = [task['task_id'] for task in new_ind2.task_array]
        assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after copy: {task_ids}"
        population.extend([new_ind1, new_ind2])

    def opt_gpu(self, population: list[individual], ind: individual):
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        # 筛选GPU任务并排序
        gpu_mask = task_array['gpu'] >= 1
        gpu_tasks = task_array[gpu_mask]
        if len(gpu_tasks) == 0:
            return
        
        sorted_indices = np.argsort(-gpu_tasks['total_runtime'])  # 倒序排序
        
        # 批量调整前1/3
        top_count = max(1, len(sorted_indices) // 3)
        top_indices = sorted_indices[:top_count]
        increments = np.random.choice([1, 2], size=top_count)
        new_gpu = np.clip(gpu_tasks[top_indices]['gpu'] + increments, 1, ind.total_resources['gpu'])
        task_array[gpu_mask][top_indices]['gpu'] = new_gpu
        
        # 批量调整后1/3
        bottom_indices = sorted_indices[-top_count:]
        decrements = np.random.choice([1], size=top_count)
        new_gpu = np.clip(gpu_tasks[bottom_indices]['gpu'] - decrements, 1, None)
        task_array[gpu_mask][bottom_indices]['gpu'] = new_gpu
        
        population.append(new_ind)

    def opt1(self, population: list, ind: individual):
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        # 按运行时间排序
        sorted_indices = np.argsort(-task_array['total_runtime'])
        
        if np.random.rand() < 0.5:  # 增加资源
            # 批量处理前1/3任务
            top_count = max(1, len(sorted_indices) // 3)
            top_indices = sorted_indices[:top_count]
            current_cpus = task_array[top_indices]['cpu']
            
            # 向量化计算增量
            # 激进
            # increments = np.select(
            #     [current_cpus < 4, current_cpus < 8],
            #     [np.random.choice([2,3,4], size=top_count),
            #     np.random.choice([4,6,8], size=top_count)],
            #     default=np.random.choice([6,8,10], size=top_count)
            # )
            # 温和
            increments = np.random.choice([1,1,2,3,4], size=top_count)
            for i, idx in enumerate(top_indices):
                max_cpu = self.node_resources[task_array[idx]['node']]['cpu']
                new_cpu = np.clip(task_array[idx]['cpu'] + increments[i], 1, max_cpu)
                task_array[idx]['cpu'] = new_cpu
        else:  # 减少资源
            # 批量处理后1/3任务
            bottom_count = max(1, len(sorted_indices) // 3)
            bottom_indices = sorted_indices[-bottom_count:]
            current_cpus = task_array[bottom_indices]['cpu']
            
            # 激进
            # decrements = np.select(
            #     [current_cpus > 8, current_cpus > 4],
            #     [np.random.choice([4,6], size=bottom_count),
            #     np.random.choice([2,3], size=bottom_count)],
            #     default=1
            # )
            # 温和
            decrements = np.random.choice([1,1,2,3,4], size=bottom_count)
            for i, idx in enumerate(bottom_indices):
                new_cpu = np.clip(task_array[idx]['cpu'] - decrements[i], 1, None)
                task_array[idx]['cpu'] = new_cpu
        
        population.append(new_ind)


    def opt2(self, population: list, ind: individual):
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        # 找到最晚完成的任务
        latest_idx = np.argmax(task_array['finish_time'])
        
        # tmp
        if latest_idx == 0:
            return
        
        # 生成新位置（避免列表操作）
        try:
            new_pos = np.random.randint(0, latest_idx)
        except ValueError:
            print(latest_idx)
            print(task_array)
        
        # 使用数组索引操作替代列表pop/insert
        indices = np.arange(len(task_array))
        mask = (indices != latest_idx)
        new_order = np.concatenate([
            indices[mask][:new_pos],
            [latest_idx],
            indices[mask][new_pos:]
        ])
        new_ind.task_array = task_array[new_order]
        new_ind.update_task_id_index()
        
        population.append(new_ind)

    def process_individual_opt(self, population):
        # logger.info(f"process_infividual:{ind1.individual_id}")
        for ind in population:
            self.opt1(ind)
            self.opt2(ind)

    def process_individual_mutate(self, population, ind1, ind2, crossover_rate=0):
        self.mutate_resources(population, ind1)
        self.mutate_resources(population, ind2)

        self.mutate_seq(population, ind1)
        self.mutate_seq(population, ind2)

        self.crossover_pmx(population, ind1, ind2)
        self.crossover_arith_ave(population, ind1, ind2)


    def clean_population(self, population):
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024**2  # in MB
        print(f"Current memory usage: {memory_usage:.2f} MB")
        for ind in population:
            del ind
        gc.collect()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024**2  # in MB
        print(f"After cleaning memory usage: {memory_usage:.2f} MB")

    def run_ga_for_node(self, node, population, num_runs_in_node, num_generations_node):
        self.write_log(f"\nNode {node} GA start")
        self.write_log(f"Initial allocation: {population[0].task_array}")

        # 检查边界条件
        if len(population) < 1 or len(population[0].task_array) < 2:
            self.write_log(f"Node {node}: No population, or only one task. Skipping...")
            return (node, population[0])

        score = 0
        new_score = 0

        for gen_node in range(num_runs_in_node):
            population = population[:num_generations_node]
            random.shuffle(population)
            size = len(population)

            for i in range(size // 2):
                ind1 = population[i]
                ind2 = population[size - i - 1]

                # if new_score == score:
                self.process_individual_mutate(population, ind1, ind2)

                self.opt1(population, ind1)
                self.opt1(population, ind2)
                self.opt2(population, ind1)
                self.opt2(population, ind2)
                self.opt_gpu(population, ind1)
                self.opt_gpu(population, ind2)

            self.sch_data.Task_time_predictor.estimate_ga_population(
                population, self.sch_data.sch_task_list, all_node=False
            )
            scores = [self.fitness(ind) for ind in population]
            population = [population[i] for i in np.argsort(scores)[::-1]]
            score = new_score
            new_score = population[0].score

            # 记录每一代的关键信息
            best_ind = population[0]
            self.write_log(
                f"Node {node} Generation {gen_node + 1}: "
                f"Score = {new_score:.2f}, "
                f"Best CPU alloc = {[task['cpu'] for task in best_ind.task_array]}, "
            )

        best_ind = max(population, key=lambda ind: ind.score)
        self.write_log(
            f"Node {node} Final Result:\n"
            f"Best Score = {best_ind.score:.2f}\n"
            f"Final Allocation = {best_ind.task_array}\n"
            f"{'-' * 80}"
        )
        return (node, best_ind)


    def update_node_tasks(self, a_ind, best_node_ind, node):
        # 直接获取指定node的任务
        node_mask = best_node_ind.task_array['node'] == node
        best_tasks = best_node_ind.task_array[node_mask]
        
        # 获取这些任务的task_ids及其在a_ind中的位置
        task_ids = best_tasks['task_id']
        positions = np.array([a_ind._task_id_index[tid] for tid in task_ids])
        
        # 按best_tasks的顺序更新这些位置的任务
        a_ind.task_array[positions] = best_tasks
        
        # 更新索引映射
        a_ind.update_task_id_index()
        
    def run_ga(
        self,
        all_tasks:list[dict[str, int]],
        num_runs: int = 10,
        num_runs_in_node: int = 50,
        num_generations_all: int = 1,
        num_generations_node: int = 50,
        pool = None,
    )->list:
        start_time = time.time()
        task_nums = self.at.get_task_nums(all_tasks)
        self.write_log(f"\nStarting GA with {task_nums} tasks, tasks list: {all_tasks}")
        self.write_log(f"Running tasks: {self.running_task_node}")
        self.write_log(f"Available resources: {self.node_resources}")
        
        # fill features from new task
        self.sch_data.Task_time_predictor.fill_features_from_new_task(self.node_resources, self.sch_data.sch_task_list)
        self.sch_data.Task_time_predictor.fill_runtime_records_with_predictor()
        self.write_log(f"Predictor filled with new task features, consuming time: {time.time() - start_time:.2f} seconds")
        
        # run no record task
        ind = self.detect_no_his_task(all_tasks)
        if ind is not None and len(ind.task_array) > 0:
            return ind.task_allocation

        self.population = self.generate_population_all(all_tasks=all_tasks, population_size=num_generations_all)
        
        predict_model_train_time = time.time()
        self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
        logger.info(f"Predictor train time: {time.time() - predict_model_train_time:.2f} seconds")
        
        self.sch_data.Task_time_predictor.estimate_ga_population(
            self.population, self.sch_data.sch_task_list, all_node=True
        )
        scores = [self.fitness(ind) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]

        # only one task , submit directly
        if self.at.get_total_nums(all_tasks) == 1:
            logger.info(f"Only one task, submit directly")
            self.best_ind = self.population[-1]  # predifined resources at last in list
            return self.best_ind.task_allocation

        score = self.population[0].score
        logger.info(f"initial score is {score}")
        new_score = 0
        # for gen in range(num_runs): ## total epoch
        #     # evo on each node， population size will influence the times for ga run
        for gen, a_ind in enumerate(self.population):
            self.write_log(f"\nGlobal Generation {gen + 1}")
            load_balance_times = 5
            num_runs = np.linspace(1, num_runs_in_node, load_balance_times).astype(int)
            for _ in range(load_balance_times):
                self.write_log(f"Before load balance: {a_ind.task_array}")
                self.load_balance(a_ind)
                self.write_log(f"After load balance: {a_ind.task_array}")
                
                self.population_node = self.generate_population_in_node(
                    a_ind, num_generations_node
                )  # generate ind on each node
                
                # 串行处理每个节点
                for node, population in self.population_node.items():
                    results = self.run_ga_for_node(
                        node, 
                        population, 
                        # num_runs_in_node, 
                        num_runs[_],
                        num_generations_node
                    )
                    node, best_ind = results
                    self.update_node_tasks(a_ind, best_ind, node)
                    self.write_log(f"Node {node} final allocation: {best_ind.task_array}")
                
                # # 并行处理每个节点
                # results = pool.starmap(
                #     self.run_ga_for_node,
                #     [
                #         (node, population, num_runs_in_node, num_generations_node)
                #         for node, population in self.population_node.items()
                #     ],
                # )
                # for node, task_allocation in results:
                #     self.write_log(f"Node {node} final allocation: {task_allocation}")

                # all node ind operation end
        # global ind operation here
        scores = [self.fitness(ind) for ind in self.population]
        # logger.info(f"Generation {gen}: best ind score:{self.population[0].score}")

        # best ind global
        best_ind = max(self.population, key=lambda ind: ind.score)
        self.best_ind = best_ind
        best_allocation = best_ind.task_allocation
        self.write_log("\nFinal Results:")
        self.write_log(f"scores of all individuals: {scores}")
        self.write_log(f"Best individual score: {best_ind.score}")
        self.write_log(f"Best allocation: {best_allocation}")
        self.write_log(f"GA running time: {time.time() - start_time:.2f} seconds")


        logger.info("GA running time: %s seconds" % (time.time() - start_time))
        # self.at.move_allocation_to_scheduled(all_tasks, best_allocation) # should consider lock
        return best_allocation