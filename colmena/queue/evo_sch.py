# Standard library imports
import uuid
import copy
import gc
import json
import logging
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

# Third-party library imports
import numpy as np
# import pandas as pd
from pandas import DataFrame
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Lazy imports for heavy ML modules
# def get_ml_models():
#     from sklearn.linear_model import Ridge, ElasticNet
#     return Ridge, ElasticNet

# Local application imports
from colmena.models import Result

# Configure logging
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


class agent_pilot:
    # 使用历史提交任务建模多项式拟合，判断那种任务适合作为下一次提交任务
    # 如果agent停在用户逻辑内部，调度器难以通过agent触发
    # 如果agent停在用户逻辑外部，调度器需要接管用户控制
    agent_dict = None

    acquire_level = None  # info for resources util

    # agent resources model {agent_name: agent_info} agent_info:[permit_submit, resources_avail]
    # resouces_level

    def __init__(self, sch_data, resources_rate, available_resources, util_level):
        # self.provision_resources = resources_rate * available_resources

        self.sch_data:Sch_data = sch_data
        self.resources_rate = resources_rate
        self.available_resources = available_resources
        self.util_level = util_level

    def build_agent_model():
        # resources model from his data
        # add new result
        # now we use average resources and time to predict
        
        pass


    # call in agent / 避免被动call 主动调用agent提供资源？ 轮询？
    def acquire_resources(self, method):
        pass


class SmartScheduler:
    # support different scheduling policy here
    
    ## init all sch model here
    # sch_data can be menber of all member model
    def __init__(self, methods, available_task_capacity, available_resources, sch_config= None):
        self.sch_data: Sch_data = Sch_data()
        # self.agent_pilot = agent_pilot(sch_data=self.sch_data, resources_rate=2, available_resources=available_resources, util_level=0.8)
        self.sch_data.init_avail_task(available_task(methods), available_task_capacity)
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
        
        # processes = len(self.node_resources)
        processes = 4
        self.pool = multiprocessing.Pool(processes=processes)
        

        hist_path = []
        hist_path.append(
            "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/test_data/simulation-results-20241224-116.json"
        )
        hist_path.append(
            "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/test_data/simulation-results-20241224-152.json"
        )

        hist_path.append(
            "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/inference-results-20240319_230707.json"
        )
        hist_path.append(
            "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/sampling-results-20241211.json"
        )
        hist_path.append(
            "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/training-results-20241211.json"
        )
        self.sch_data.historical_task_data.get_features_from_his_json(hist_path)
        # self.sch_data.Task_time_predictor.polynomial_train(self.sch_data.historical_task_data.historical_data)
        
        logger.info('init smart scheduler')
        
    def __del__(self):
        self.pool.close()
        self.pool.join()
        
    
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
        
    def run_sch(self, method = "mrsa", model_type="powSum"):
        """运行调度器

        Args:
            method (str, optional): 选择ga或mrsa调度器. Defaults to "ga".
            model_type (str, optional): 选择mrsa时设置amdMax, amdSum, powMax, powSum四种性能模型. Defaults to "powSum".

        Returns:
            _type_: _description_
        """
        if method == "ga":
        # run evo sch
            with self.sch_lock:
                all_tasks = self.sch_data.avail_task.get_all()
                all_tasks = copy.deepcopy(all_tasks)
                best_allocation = self.evo_sch.run_ga(all_tasks, pool = self.pool)
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
        cmd = f"python3 /home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/alphaMaster.py {folder_name} 2 {total_cpu} {total_gpu} {model_type} {folder_name}"
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
            y_gpu = np.array(times_gpu) - s0 - s1/optimal_cpu
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
                y_gpu = np.log(np.array(times_gpu) - serial_time - s1/(optimal_cpu**a1))
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
    os.makedirs(f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}", exist_ok=True)
    output_file = f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}/sample0.txt"
    
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
    
    if sch_data.Task_time_predictor.model_type == "random_forest":
        models = sch_data.Task_time_predictor.random_forest_models
    elif sch_data.Task_time_predictor.model_type == "polynomial":
        models = sch_data.Task_time_predictor.polynomial_models
    # 转换为power-max格式
    convert_to_mrsa_models(
        task_list=tasks,
        models=models,
        model_type=model_type,
        output_folder=output_folder
    )
    
    # 创建空的依赖关系文件夹和文件(因为不考虑依赖关系)
    os.makedirs(f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}", exist_ok=True)
    with open(f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}/sample0.txt", 'w') as f:
        pass
    
    # 创建分配结果文件夹
    os.makedirs(f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/allocation/{output_folder}", exist_ok=True)
    
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
    with open(f"/home/lizz_lab/cse12232433/project/colmena/multisite_/mrsa/files/allocation/{output_folder}/sample0seq.txt", 'r') as f:
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
    tasks_nums: Dict[str, int] = field(default_factory=dict)
    total_resources: dict = field(default_factory=dict)
    total_time: int = 0
    max_time: int = 0
    score: int = 0

    ## optional
    predict_run_seq: list = field(default_factory=list)  # [task,,,]
    predict_run_seq_node: dict = field(
        default_factory=dict
    )  # {'node':{'task_id':task,,,},,,}

    # allocation information
    task_allocation: list[dict[str, int]] = field(
        default_factory=list
    )  # store all task
    task_allocation_node: dict = field(default_factory=dict)  # store task on each node

    # initialize individual id
    def __post_init__(self):
        if self.individual_id == -1:
            self.individual_id = individual._next_id
            individual._next_id += 1

    # deepcopy individual
    def copy(self):
        copied_individual = copy.deepcopy(self)
        copied_individual.individual_id = individual._next_id
        individual._next_id += 1
        return copied_individual

    # hash
    def __hash__(self) -> int:
        sorted_allocation = sorted(
            self.task_allocation, key=lambda x: (x['name'], x['task_id'])
        )
        return hash(str(sorted_allocation))

    def get_task_resources(self, task_name, task_id):
        for task in self.task_allocation:
            if task['name'] == task_name and task['task_id'] == task_id:
                return task['resources']
        return None

    # convert to json
    def to_json(self):
        return json.dumps(dataclass_to_dict(self), indent=4)

    # save to json file
    def save_to_json(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(dataclass_to_dict(self), f, indent=4)


class Sch_data(SingletonClass):
    def __init__(self):
        self.result_list = {}
        self.sch_task_list = {}
        self.pilot_task = {}
        self.Task_time_predictor: TaskTimePredictor = None
        self.avail_task: available_task = None
        self.avail_task_cap: int = None

    def init_hist_task(self, historical_task_data):
        self.historical_task_data:HistoricalData = historical_task_data

    def init_avail_task(self, avail_task_data, avail_task_cap):
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


class Avail_task(SingletonClass):
    # only save useful data from features in result object
    # methods
    # task_id
    # resources
    # input size
    # some input paremeters defined by user
    def __init__(self):
        pass

    # maybe we dont need develop this class


@dataclass
class available_task(SingletonClass):
    # task_names: list[str] = field(default_factory=list)
    # task_ids: list[dict[str, int]] = field(default_factory=dict)

    # def __init__(self,task_names: set[str], task_ids: dict[str, int], task_datas=None):
    # def __init__(self, task_ids: dict[str, list[str]] = None):
    def __init__(self, task_methods: list[str]):
        self.task_ids = {method: [] for method in task_methods}

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

    def remove_task_id(self, task_name: str, task_id: Union[str, list[str]]):
        ## judge if there is task id
        task_id = [task_id] if isinstance(task_id, str) else task_id

        for i in task_id:
            if i not in self.task_ids[task_name]:
                print(f"task id {i} not in task name {task_name}")
                # logging.warning(f"task id {task_id} not in task name {task_name}")
                continue
            self.task_ids[task_name].remove(i)
            # if len(self.task_ids[task_name]) == 0:
            #     self.task_ids.pop(task_name)
        # if len(self.task_ids[task_name]) == 0:
        #     # print type
        #     print(type(self.task_names))
        #     print(self.task_names)
        #     print(task_name)
        #     self.task_names.remove(task_name)

    def get_available_task_id(self, task_name):
        return self.task_ids.get(task_name)

    def get_all(self):
        return self.task_ids

    def get_task_nums(self, all_tasks):
        result = {}
        for key, value in all_tasks.items():  # key: task name, value:task id list
            result[key] = len(value)
        return result

    def get_total_nums(self, all_tasks = None):
        if all_tasks == None:
            all_tasks = self.get_all()
        return sum(self.get_task_nums(all_tasks).values())


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
                    
    # def calculate_task_average_metrics(self):
    #     """ Calculate the average resources and runtime for each method. """
    #     # because no linear performance; should improved
    #     averages = {}
    #     for method, data in self.historical_data.items():
    #         if not data:
    #             continue  # Skip if there's no data for the method
            
    #         total_cpu = total_gpu = total_time = 0
    #         count = len(data)

    #         for entry in data:
    #             total_cpu += entry.get("resources.cpu", 0)
    #             total_gpu += entry.get("resources.gpu", 0)
    #             total_time += entry.get("time_running", 0)

    #         averages[method] = {
    #             "average_cpu": total_cpu / count,
    #             "average_gpu": total_gpu / count,
    #             "average_time": total_time / count,
    #         }

    #     return averages


class TaskTimePredictor:
    random_forest_models: dict[str, RandomForestRegressor] = field(default_factory=dict)
    polynomial_models: dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, methods, features, model_type: Literal["random_forest", "polynomial"] = "random_forest"):
        """
        初始化任务时间预测器
        
        Args:
            methods: 方法列表
            features: 特征列表
            model_type: 模型类型，可选 "random_forest" 或 "polynomial"
        """
        self.model_type = model_type
        self.features = features
        self.random_forest_models = {}
        self.polynomial_models = {}
        
        # 初始化模型
        for method in methods:
            if model_type == "polynomial":
                self.random_forest_models[method] = None
            else:
                self.polynomial_models[method] = None

    def train(self, train_data):
        """统一的训练入口"""
        if self.model_type == "random_forest":
            self.random_forest_train(train_data)
        elif self.model_type == "polynomial":
            self.polynomial_train(train_data)

    def random_forest_train(self, train_data):
        for method, data in train_data.items():
            df = DataFrame(data)
            df.dropna(inplace=True)
            
            if len(data) < 5:
                continue

            x = df.drop(columns=['time_running', 'method', 'task_id'])
            y = df['time_running']
            
            y = np.log1p(y)
            
            rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='auto',
                    n_jobs=-1,
                    random_state=42
                ))
            ])
            
            rf_model.fit(x, y)
            self.random_forest_models[method] = rf_model

    def polynomial_train(self, train_data):
        for method, data in train_data.items():
            df = DataFrame(data)
            df.dropna(inplace=True)
            if len(data) < 5:
                continue

            x = df.drop(columns=['time_running', 'method', 'task_id'])
            y = df['time_running']

            poly_model = Pipeline([
                ('scaler', StandardScaler()),  # 添加标准化步骤
                ('poly_features', PolynomialFeatures(degree=3)),
                ('linear_model', LinearRegression()),
            ])
            
            y = np.log1p(y)
            poly_model.fit(x, y)
            self.polynomial_models[method] = poly_model

    def estimate_time(self, task):
        """
        预测单个任务的运行时间
        
        param:
        task: 通过get_feature_from_task获取的任务特征
        """
        method = task['method']
        X = DataFrame([task]).drop(columns=['time_running', 'method', 'task_id'])
        
        if self.model_type == "random_forest":
            model = self.random_forest_models.get(method)
            if model is None:
                return None
            prediction = model.predict(X)[0]
        elif self.model_type == "polynomial":
            model = self.polynomial_models.get(method)
            if model is None:
                return None
            prediction = model.predict(X)[0]
            
        return np.expm1(prediction)

    def extract_feature_from_task(self, task: Result, result: Result):
        task_features = {}
        for feature in self.features['default']:
            value = result
            for key in feature.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = getattr(value, key)
            if key in task['resources']:
                value = task['resources'][key]
            task_features[feature] = value
        return task_features

    def estimate_ga_population(self, population, sch_task_list, all_node=False):
        tasks_by_model = {}
        for ind in population:
            if all_node:
                task_allocation = []
                for key in ind.task_allocation_node.keys():
                    task_allocation.extend(ind.task_allocation_node[key])
            else:
                task_allocation = ind.task_allocation

            for task in task_allocation:
                model_name = task['name']
                if model_name not in tasks_by_model:
                    tasks_by_model[model_name] = []
                tasks_by_model[model_name].append(task)

        for model_name, tasks in tasks_by_model.items():
            if self.model_type == "random_forest":
                model = self.random_forest_models.get(model_name)
            elif self.model_type == "polynomial":
                model = self.polynomial_models.get(model_name)
                
            if model is None:
                continue

            feature_values = []
            for task in tasks:
                feature = copy.deepcopy(sch_task_list[task['task_id']])
                feature['resources.cpu'] = task['resources']['cpu']
                feature['resources.gpu'] = task['resources']['gpu']
                feature_values.append(feature)

            X = DataFrame(feature_values).drop(columns=['time_running', 'method', 'task_id'])
            predictions = model.predict(X)
            predictions = np.expm1(predictions)
            
            for task, runtime in zip(tasks, predictions):
                task['total_runtime'] = runtime


# @dataclass
# class historical_data(
#     SingletonClass
# ):  # change to sch data, contain all data and transfer to child class in queue / thinker
#     features = [
#         "method",
#         "message_sizes.inputs",
#         # "worker_info.hostname", # for multinode executor # need vectorize for training
#         "resources.cpu",
#         "resources.gpu",
#         # "resources.thread",
#         "time_running",
#     ]

#     def __init__(self, methods: Collection[str], queue=None):
#         # submit history and complete history for cal the potential improvement
#         self.methods = methods
#         self.submit_task_seq = []
#         self.complete_task_seq = []
#         ## total submit / total complete
#         self.trigger_info = []

#         # his data for predict time runnint
#         self.historical_data = {}
#         self.random_forest_model = {}
#         from sklearnex import patch_sklearn, unpatch_sklearn

#         patch_sklearn()
#         for method in self.methods:
#             self.historical_data[method] = []
#             self.random_forest_model[method] = RandomForestRegressor(
#                 n_estimators=100, random_state=42, n_jobs=-1
#             )

#         self.queue = queue

#     def add_data(self, feature_values: dict[str, Any]):
#         method = feature_values['method']
#         # if method not in self.historical_data:
#         #     self.historical_data[method] = []
#         #     self.random_forest_model[method] = RandomForestRegressor(n_estimators=100, random_state=42)
#         if method not in self.historical_data:
#             logger.warning(f"method {method} not in historical data")
#         self.historical_data[method].append(feature_values)

#     def get_features_from_result_object(self, result: Result):
#         feature_values = {}
#         for feature in self.features:
#             value = result
#             for key in feature.split('.'):
#                 if isinstance(value, dict):
#                     value = value.get(key)
#                     break
#                 else:
#                     value = getattr(value, key)
#             if feature == 'resources.gpu':
#                 value = (
#                     len(value) if isinstance(value, list) else 0
#                 )  # 如果是list，则取长度，否则默认为0
#                 # if value is None:
#                 #     break
#             feature_values[feature] = value
#         self.add_data(feature_values)

#     def get_features_from_his_json(self, his_json: Union[str, list[str]]):
#         for path in his_json:
#             with open(path, 'r') as f:
#                 for line in f:
#                     json_line = json.loads(line)
#                     feature_values = {}
#                     for feature in self.features:
#                         value = json_line
#                         for key in feature.split('.'):
#                             value = value.get(key)
#                             # if value is None:
#                             #     break
#                         feature_values[feature] = value
#                     self.add_data(feature_values)

#     def random_forest_train(self):
#         for method in self.historical_data:
#             logger.info(f"train:{method} model")
#             data = self.historical_data[method]
#             df = DataFrame(data)
#             df.dropna(inplace=True)
#             if len(data) < 5:
#                 continue
#             model = self.random_forest_model[method]

#             X = []
#             y = []
#             for feature_values in data:
#                 X = df.drop(columns=['time_running', 'method', 'task_id'])
#                 y = df['time_running']
#             # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1.0, random_state=42)
#             model.fit(X, y)
#             logger.info(
#                 f"method: {method}, random forest regressor score: {model.score(X, y)}"
#             )
#             # print(f"method: {method}, random forest regressor score: {model.score(X, y)}")

#     def estimate_time(self, task):
#         # use allocate resource to estimate running time
#         method = task['name']
#         result: Result = self.queue.result_list[task['task_id']]
#         model = self.random_forest_model[method]
#         feature_values = {}
#         for feature in self.features:
#             value = result
#             for key in feature.split('.'):
#                 if isinstance(value, dict):
#                     value = value.get(key)
#                     break
#                 else:
#                     value = getattr(value, key)
#             if key in task['resources']:
#                 value = task['resources'][key]
#             feature_values[feature] = value
#         X = DataFrame([feature_values]).drop(columns=['time_running', 'method', 'task_id'])
#         return model.predict(X)[0]

#     def estimate_batch(self, population, all_node=False):
#         def extract_feature_values(task, result):
#             feature_values = {}
#             for feature in self.features:
#                 value = result
#                 for key in feature.split('.'):
#                     if isinstance(value, dict):
#                         value = value.get(key)
#                         break
#                     else:
#                         value = getattr(value, key)
#                 if key in task['resources']:
#                     value = task['resources'][key]
#                 feature_values[feature] = value
#             return feature_values

#         tasks_by_model = {}  # 以模型为键，将任务分组存储
#         for ind in population:
#             if all_node:
#                 task_allocation = []
#                 for key in ind.task_allocation_node.keys():
#                     task_allocation.extend(ind.task_allocation_node[key])
#             else:
#                 task_allocation = ind.task_allocation

#             for task in task_allocation:
#                 model_name = task['name']
#                 if model_name not in tasks_by_model:
#                     tasks_by_model[model_name] = []
#                 tasks_by_model[model_name].append(task)

#         for model_name, tasks in tasks_by_model.items():
#             model = self.random_forest_model[model_name]  # 获取对应的模型

#             feature_values = []
#             for task in tasks:
#                 result: Result = self.queue.result_list[task['task_id']]
#                 feature_dict = extract_feature_values(task, result)
#                 feature_values.append(feature_dict)

#             X = DataFrame(feature_values).drop(columns=['time_running', 'method', 'task_id'])
#             predictions = model.predict(X)

#             for task, runtime in zip(tasks, predictions):
#                 task['total_runtime'] = runtime
#                 # logger.info(f"Predicted runtime for task {task['task_id']}: {runtime}")


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
        self.running_task = []
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
            # setattr(result_obj.resources, 'cpu', cpu_value)
            # setattr(result_obj.resources, 'gpu', gpu_value)
            result_obj.inputs[1]['gpu'] = gpu_value
            result_obj.inputs[1]['cpu'] = cpu_value
            # self.at.remove_task_id(task_name=name, task_id=task_id)
    
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

class Back_Filing_Scheduler:
    pass

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
        self.log_dir = "/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/job_out"
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
            self.at.remove_task_id(task_name=result_obj['name'], task_id=result_obj['task_id'])

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

    def ind_init_task_allocation(self):
        """specify task_allocation_node formate = {node: [task,,,],,,}

        Returns:
            _type_: _description_
        """
        task_allocation_node = {}
        for node in self.node_resources.keys():
            task_allocation_node[node] = []
        return task_allocation_node

    def check_pending_task_on_node(self, task_allocation):
        if task_allocation is None:
            return True  # no ind means no pending task

        # check if there is pending task on node, no pending task trigger evosch
        pending_queue = self.ind_init_task_allocation()
        for task in task_allocation:
            node = task['resources']['node']
            pending_queue[node].append(task)
        for node in pending_queue:
            if len(pending_queue[node]) == 0:  # no pending task on node
                return True

        return False  # all node have task pending

    def get_piority(self):
        pass

    def cpu_choice(self):
        pass

    def gpu_choice(self):
        pass

    def detect_submit_sequence(self):
        # detect submit task sequence to choose proper task to run
        pass

    def detect_no_his_task(self, all_tasks, total_nums=5):
        '''
        detect if there is task not in historical data
        without historical data, estimate method cannot work, we need run them first, and record the historical data to train the model
        return the individual contain no record task
        # get all node resource or get user predefine resource
        '''
        if all_tasks == None:
            all_tasks = {}
        task_queue = defaultdict(list)
        predict_running_seq = defaultdict(list)
        # all_tasks = self.at.get_all()
        which_node = self.generate_node()
        for name, ids in all_tasks.items():
            avail = len(ids)
            hist = len(
                # self.hist_data.historical_data[name]
                self.sch_data.historical_task_data.historical_data[name]
            )  # we dont have historical data for estimate
            if (hist < total_nums) and (avail > 0):
                # cpu choices
                # predefine_cpu = getattr(
                #     # self.hist_data.queue.result_list[ids[0]].resources, 'cpu'
                #     self.sch_data.result_list[ids[0]].resources, 'cpu'
                # )
                # logger.info(f"predefine cpu: {predefine_cpu}")
                predefine_cpu = self.sch_data.sch_task_list[ids[0]]['resources.cpu']
                # logger.info(f"predefine cpu: {predefine_cpu}")
                cpu_lower_bound = min(2, predefine_cpu // 2)
                cpu_upper_bound = max(
                    self.node_resources.values(), key=lambda x: x['cpu']
                )[
                    'cpu'
                ]  # node max cpu value
                sample_nums = min((total_nums - hist), avail)
                choices = np.linspace(
                    cpu_lower_bound,
                    cpu_upper_bound,
                    num=sample_nums,
                    endpoint=True,
                    retstep=False,
                    dtype=int,
                )

                # gpu choices
                # predefine_gpu = getattr(
                #     # self.hist_data.queue.result_list[ids[0]].resources, 'gpu'
                #     self.sch_data.result_list[ids[0]].resources, 'gpu'
                # )
                # logger.info(f"predefine gpu: {predefine_gpu}")
                predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
                # logger.info(f"predefine gpu: {predefine_gpu}")
                if predefine_gpu == 0:  # this task do not use gpu
                    gpu_choices = np.linspace(
                        0, 0, num=sample_nums, endpoint=True, retstep=False, dtype=int
                    )
                else:
                    gpu_lower_bound = 1
                    gpu_upper_bound = max(
                        self.node_resources.values(), key=lambda x: x['gpu']
                    )['gpu']
                    gpu_choices = np.linspace(
                        gpu_lower_bound,
                        gpu_upper_bound,
                        num=sample_nums,
                        endpoint=True,
                        retstep=False,
                        dtype=int,
                    )
                # if len(self.hist_data.historical_data[name])<5: # no data for train
                for i in range(len(choices)):
                    cpu = int(choices[i])
                    gpu = int(gpu_choices[i])
                    node = next(which_node)
                    new_task = {
                        "name": name,
                        "task_id": ids[i],
                        "resources": {"cpu": cpu, "gpu": gpu, "node": node},
                        "total_runtime": 1,
                    }
                    task_queue[node].append(new_task)
                    predict_running_seq[node].append(
                        {
                            'name': name,
                            'task_id': ids[i],
                            'start_time': None,
                            'finish_time': None,
                            'total_runtime': 1,  # 无意义值
                            'resources': {'cpu': cpu, 'gpu': gpu, "node": node},
                        }
                    )
        ind = individual(
            tasks_nums=self.get_dict_list_nums(task_queue),
            total_resources=self.node_resources,
        )
        ind.task_allocation_node = task_queue
        task_allocation = []
        for key in task_queue.keys():
            task_allocation.extend(task_queue[key])
        ind.task_allocation = task_allocation
        ind.predict_run_seq_node = predict_running_seq
        return ind

    def calc_time(self, tasks, node):
        total_cpu_time = 0
        total_gpu_time = 0
        for task in tasks:
            total_cpu_time += task['resources']['cpu'] * task['total_runtime']
            total_gpu_time += task['resources']['gpu'] * task['total_runtime']
        return total_cpu_time, total_gpu_time

    def calc_utilization(self, ind):
        total_cpu_time = defaultdict(int)
        total_gpu_time = defaultdict(int)
        completion_time = defaultdict(int)
        for node in self.node_resources.keys():
            total_cpu_time[node], total_gpu_time[node] = self.calc_time(
                ind.task_allocation_node[node], node
            )
            completion_time[node], running_swq = (
                self.calculate_completion_time_record_with_running_task(
                    self.node_resources[node],
                    self.running_task_node[node],
                    ind.task_allocation_node[node],
                )
            )

        return total_cpu_time, total_gpu_time, completion_time

    def load_balance(self, ind):
        # 不考虑通信迁移成本，只考虑每个节点的完成时间
        if not isinstance(ind, individual):
            raise ValueError("load_balance input is not individual")

        # TODO consider resources constraint in each node, prevent resources not enough
        total_cpu_time, total_gpu_time, completion_time = self.calc_utilization(ind)
        diff_cur = max(completion_time.values()) - min(completion_time.values())
        diff_pre = max(completion_time.values())
        max_pre = max(completion_time.values())
        max_node = max(completion_time, key=completion_time.get)
        min_node = min(completion_time, key=completion_time.get)
        while diff_cur < diff_pre:
            max_node = max(completion_time, key=completion_time.get)
            min_node = min(completion_time, key=completion_time.get)

            best_task = None
            best_diff = float('inf')

            for task in ind.task_allocation_node[max_node]:
                used_cpu_time_max = (
                    total_cpu_time[max_node]
                    - task['resources']['cpu'] * task['total_runtime']
                )
                total_cpu_area_max = completion_time[max_node] * self.resources_evo[max_node]['cpu']

                temp_gpu_time_max = (
                    total_gpu_time[max_node]
                    - task['resources']['gpu'] * task['total_runtime']
                )
                total_gpu_area_max = completion_time[max_node] * self.resources_evo[max_node]['gpu']

                temp_cpu_time_min = (
                    total_cpu_time[min_node]
                    + task['resources']['cpu'] * task['total_runtime']
                )
                total_cpu_area_min = completion_time[min_node] * self.resources_evo[min_node]['cpu']

                temp_gpu_time_min = (
                    total_gpu_time[min_node]
                    + task['resources']['gpu'] * task['total_runtime']
                ) 
                total_gpu_area_min = completion_time[min_node] * self.resources_evo[min_node]['gpu']

                # cpu 和 gpu 时间差的绝对值， 存疑； 应该修改为最能减少completion time； 是否应该再用一次GA算法？
                # ga on node
                # calculate completion time, or extra metrics
                # temp_diff_max = abs(temp_cpu_time_max - temp_gpu_time_max)
                # temp_diff_min = abs(temp_cpu_time_min - temp_gpu_time_min)
                # combined_diff = temp_diff_max + temp_diff_min
                temp_max_diff = abs(total_cpu_area_max - used_cpu_time_max) + abs(total_gpu_area_max - temp_gpu_time_max)
                temp_min_diff = abs(total_cpu_area_min - temp_cpu_time_min) + abs(total_gpu_area_min - temp_gpu_time_min)
                combined_diff = temp_max_diff + temp_min_diff

                if combined_diff < best_diff:
                    best_diff = combined_diff
                    best_task = task

            if best_task:
                ind.task_allocation_node[max_node].remove(best_task)
                ind.task_allocation_node[min_node].append(best_task)
                total_cpu_time[max_node] -= (
                    best_task['resources']['cpu'] * best_task['total_runtime']
                )
                total_gpu_time[max_node] -= (
                    best_task['resources']['gpu'] * best_task['total_runtime']
                )
                total_cpu_time[min_node] += (
                    best_task['resources']['cpu'] * best_task['total_runtime']
                )
                total_gpu_time[min_node] += (
                    best_task['resources']['gpu'] * best_task['total_runtime']
                )
                best_task['resources']['node'] = min_node
                completion_time[max_node], _ = (
                    self.calculate_completion_time_record_with_running_task(
                        self.node_resources[max_node],
                        self.running_task_node[max_node],
                        ind.task_allocation_node[max_node],
                    )
                )
                completion_time[min_node], _ = (
                    self.calculate_completion_time_record_with_running_task(
                        self.node_resources[min_node],
                        self.running_task_node[min_node],
                        ind.task_allocation_node[min_node],
                    )
                )
                diff_pre = diff_cur
                max_now = max(completion_time.values())
                diff_cur = max(completion_time.values()) - min(completion_time.values())
            else:
                break

            if max_now > max_pre:
                # if the max completion time increase, break
                break
            else:
                max_pre = max_now

    def load_balance_by_area():
        pass

    def calculate_completion_time_record_with_running_task(
        self, resources, running_task: list, task_allocation
    ):
        current_time = time.time()  # time line move
        record = current_time  # start time
        ongoing_task = []  # tmp use here
        running_seq = (
            {}
        )  # log seq and time information, supply for evo to choose which task to opt or mutate
        # available_resources = {node: {'cpu': res['cpu'], 'gpu': res['gpu']} for node, res in self.node_resources.items()}
        available_resources = copy.deepcopy(resources)

        # add running task
        if running_task:
            for task in running_task:
                heapq.heappush(
                    ongoing_task,
                    (
                        task['finish_time'],
                        task['resources']['cpu'],
                        task['resources']['gpu'],
                        task['task_id'],
                        task['name'],
                        task['resources']['node'],
                    ),
                )
                available_resources['cpu'] -= task['resources']['cpu']
                available_resources['gpu'] -= task['resources']['gpu']
                running_seq[task['task_id']] = {
                    'name': task['name'],
                    'task_id': task['task_id'],
                    'start_time': task['start_time'],
                    'finish_time': task['finish_time'],  # refresh at finish
                    'total_runtime': task['total_runtime'],  # refresh at finish
                    'resources': task['resources'],
                }

        for task in task_allocation:
            # start a new task
            node = task['resources']['node']
            required_cpu = task['resources']['cpu']
            required_gpu = task['resources']['gpu']

            # before add each task, check if any task completed
            while ongoing_task and ongoing_task[0][0] <= current_time:
                _, cpus, gpus, finished_task_id, task_name, task_node = heapq.heappop(
                    ongoing_task
                )
                available_resources['cpu'] += cpus
                available_resources['gpu'] += gpus
                task_record = running_seq[finished_task_id]
                task_record['finish_time'] = current_time
                task_record['total_runtime'] = current_time - task_record['start_time']

            # wait for task release resources
            while (
                available_resources['cpu'] < required_cpu
                or available_resources['gpu'] < required_gpu
            ):
                if ongoing_task:
                    (
                        next_finish_time,
                        cpus,
                        gpus,
                        finished_task_id,
                        task_name,
                        task_node,
                    ) = heapq.heappop(ongoing_task)
                    task_record = running_seq[finished_task_id]
                    task_record['finish_time'] = current_time
                    task_record['total_runtime'] = (
                        current_time - task_record['start_time']
                    )
                    current_time = next_finish_time  # time move
                    available_resources['cpu'] += cpus
                    available_resources['gpu'] += gpus
                else:
                    # all task release, resources still not enough
                    raise ValueError("Not enough resources for all tasks")

            available_resources['cpu'] -= required_cpu
            available_resources['gpu'] -= required_gpu
            start_time = current_time
            finish_time = current_time + task['total_runtime']
            heapq.heappush(
                ongoing_task,
                (
                    finish_time,
                    required_cpu,
                    required_gpu,
                    task['task_id'],
                    task['name'],
                    node,
                ),
            )

            running_seq[task['task_id']] = {
                'name': task['name'],
                'task_id': task['task_id'],
                'start_time': start_time,
                'finish_time': None,
                'total_runtime': None,
                'resources': task['resources'],
            }

        # time move on and log task complete
        while ongoing_task:
            next_finish_time, cpus, gpus, finished_task_id, task_name, task_node = (
                heapq.heappop(ongoing_task)
            )
            available_resources['cpu'] += cpus
            available_resources['gpu'] += gpus
            current_time = next_finish_time
            task_record = running_seq[finished_task_id]
            task_record['finish_time'] = current_time
            task_record['total_runtime'] = current_time - task_record['start_time']

        # 返回总完成时间和任务运行序列
        return current_time - record, running_seq

    def calculate_total_time(self, ind: individual):
        total_time = 0
        for task in ind.task_allocation:
            # total_time += self.hist_data.estimate_time(task)
            total_time += task['total_runtime']
        return total_time

    def fitness(self, ind: individual, all_node=False):
        if all_node:
            # calculate total time based on avail resources and task
            total_time = 0  ## time accumulate by all task
            # completion_time = [ 0 for _ in range(len(self.node_resources.keys()))] ## HPC makespan
            completion_time = defaultdict(list)

            total_time = self.calculate_total_time(ind)
            for node in self.node_resources.keys():
                completion_time[node], ind.predict_run_seq_node[node] = (
                    self.calculate_completion_time_record_with_running_task(
                        self.node_resources[node],
                        self.running_task_node[node],
                        ind.task_allocation_node[node],
                    )
                )

            # ind.score = -completion_time
            ind.score = -max(completion_time.values())
            return ind.score

        else:
            total_time = 0
            completion_time = 0
            completion_time, ind.predict_run_seq = (
                self.calculate_completion_time_record_with_running_task(
                    ind.total_resources,
                    self.running_task_node[ind.task_allocation[0]['resources']['node']],
                    ind.task_allocation,
                )
            )

            ind.score = -completion_time
            return ind.score

    def generate_node(self):
        nodes: list = list(self.node_resources.keys())
        index = 0
        while True:
            yield nodes[index]
            index = (index + 1) % len(nodes)

    def generate_population_all(self, all_tasks, population_size: int):
        ## 把每个任务轮转的放在每个节点上。

        which_node = self.generate_node()
        ## add all task to individual
        task_nums = self.at.get_task_nums(all_tasks)
        # all_tasks = self.at.get_all()
        population = []
        cpu_upper_bound = min(self.node_resources.values(), key=lambda x: x['cpu'])[
            'cpu'
        ]
        cpu_range = min(
            16, cpu_upper_bound
        )
        gpu_upper_bound = max(self.node_resources.values(), key=lambda x: x['gpu'])[
            'gpu'
        ]
        gpu_range = min(4, gpu_upper_bound)  # should consider node GPU resources

        # TODO tmp test, resources range should determine by node
        # generate random resources for individual
        for _ in range(population_size):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=self.node_resources)
            task_queue = self.ind_init_task_allocation() # modify to multi node, dict{'node_name':[],'node_name':[]}
            for name, ids in all_tasks.items():
                if len(ids)==0:
                    continue
                predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
                for task_id in ids:
                    node = next(which_node)
                    new_task = {
                        "name":name,
                        "task_id": task_id,
                        "resources":{
                            "cpu": random.randint(1,cpu_range),
                            "gpu": random.randint(1,gpu_range) if predefine_gpu>0 else 0, # 0 determin this task do not need gpu
                            "node": node
                        }
                    }
                    task_queue[node].append(new_task)
            for key in task_queue.keys():
                random.shuffle(task_queue[key])

            ind.task_allocation_node = task_queue
            task_allocation = []
            for key in task_queue.keys():
                task_allocation.extend(task_queue[key])
            ind.task_allocation = task_allocation
            population.append(ind)

        # # initial resources minimum
        ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=self.node_resources)
        task_queue = self.ind_init_task_allocation()
        for name, ids in all_tasks.items():
            if len(ids)==0:
                continue
            predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']
            for task_id in ids:
                node = next(which_node)
                new_task = {
                    "name":name,
                    "task_id": task_id,
                    "resources":{
                        # "cpu": random.randint(1,16)
                        "cpu": 1,
                        "gpu": min(1,predefine_gpu), # zero or one
                        "node": node
                    }
                }
                task_queue[node].append(new_task)
        for key in task_queue.keys():
            random.shuffle(task_queue[key])
        ind.task_allocation_node = task_queue
        task_allocation = []
        for key in task_queue.keys():
            task_allocation.extend(task_queue[key])
        ind.task_allocation = task_allocation
        population.append(ind)

        ## initial resources predifine
        ind = individual(
            tasks_nums=copy.deepcopy(task_nums),
            total_resources=self.node_resources,
        )
        task_queue = self.ind_init_task_allocation()
        for name, ids in all_tasks.items():
            if len(ids) == 0:
                continue

            predefine_cpu = self.sch_data.sch_task_list[ids[0]]['resources.cpu']
            predefine_gpu = self.sch_data.sch_task_list[ids[0]]['resources.gpu']

            for task_id in ids:
                node = next(which_node)
                new_task = {
                    "name": name,
                    "task_id": task_id,
                    "resources": {
                        "cpu": predefine_cpu,
                        "gpu": predefine_gpu,
                        "node": node,
                    },
                }
                task_queue[node].append(new_task)
        for key in task_queue.keys():
            random.shuffle(task_queue[key])

        ind.task_allocation_node = task_queue
        task_allocation = []
        for key in task_queue.keys():
            task_allocation.extend(task_queue[key])
        ind.task_allocation = task_allocation
        population.append(ind)

        # self.population = population
        return population

    def generate_population_in_node(self, ind: individual, pop_size: int = 10):
        """generate population in each node"""
        population_node = defaultdict(list)
        task_allocation_node = ind.task_allocation_node
        for node in self.node_resources.keys():
            task_nums = len(task_allocation_node[node])
            if task_nums == 0:
                continue
            for i in range(pop_size):
                n_ind = individual(
                    tasks_nums=copy.deepcopy(task_nums),
                    total_resources=self.node_resources[node],
                )
                n_ind.task_allocation = copy.deepcopy(task_allocation_node[node])
                random.shuffle(n_ind.task_allocation)  # shuffle task run seq
                population_node[node].append(n_ind)

            # use fitness function set pred run seq evo operation need it
            scores = [
                self.fitness(ind, all_node=False) for ind in population_node[node]
            ]

        return population_node

    def mutate_cpu(self, population: list, ind_input: individual):
        ind = ind_input.copy()
        ## change resource
        alloc = random.choice(ind.task_allocation)
        choice = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
        new_alloc = alloc['resources']['cpu'] + random.choice(choice)

        if new_alloc <= 0:
            alloc['resources']['cpu'] = 1
        elif new_alloc >= self.node_resources[alloc['resources']['node']]['cpu']:
            new_alloc = self.node_resources[alloc['resources']['node']]['cpu']
        else:
            alloc['resources']['cpu'] = new_alloc

        population.append(ind)

    def mutate_seq(self, population: list, ind_input: individual):
        ind = ind_input.copy()
        ## change task sequence
        index1 = random.randrange(len(ind.task_allocation))
        index2 = random.randrange(len(ind.task_allocation))
        while index2 == index1:
            index2 = random.randrange(len(ind.task_allocation))

        ind.task_allocation[index1], ind.task_allocation[index2] = (
            ind.task_allocation[index2],
            ind.task_allocation[index1],
        )

        population.append(ind)

    def crossover_arith_ave(
        self, population: list, ind_input1: individual, ind_input2: individual
    ):
        ind1 = ind_input1.copy()
        # task_avg = [None]*len(ind1.task_allocation)
        # for i in range(len(ind1.task_allocation)):
        #     name = ind1.task_allocation[i]['name']
        #     task_id = ind1.task_allocation[i]['task_id']
        #     task_avg[i] = {
        #         "name": name,
        #         "task_id": task_id,
        #         "resources":{
        #             "cpu": (ind1.get_task_resources(name,task_id)['cpu']+ind2.get_task_resources(name,task_id)['cpu'])//2
        #     }}
        # ind1.task_allocation = task_avg
        for task in ind1.task_allocation:
            name = task['name']
            task_id = task['task_id']
            task['resources']['cpu'] = (
                ind_input1.get_task_resources(name, task_id)['cpu']
                + ind_input2.get_task_resources(name, task_id)['cpu']
            ) // 2
        population.append(ind1)

    def list_dict_found(self, list_dic, dic):
        for i in range(len(list_dic)):
            if (
                list_dic[i]['task_id'] == dic['task_id']
                and list_dic[i]['name'] == dic['name']
            ):
                return True
        return False

    def list_dict_index(self, list_dic, dic):
        for i in range(len(list_dic)):
            if (
                list_dic[i]['task_id'] == dic['task_id']
                and list_dic[i]['name'] == dic['name']
            ):
                return i
        return None

    def crossover_pmx(
        self, population: list, ind_input1: individual, ind_input2: individual
    ):
        ind1 = ind_input1.copy()
        ind2 = ind_input2.copy()
        size = len(ind1.task_allocation)
        p1, p2 = [0] * size, [0] * size

        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # print(cxpoint1,cxpoint2)
        for i in range(cxpoint1, cxpoint2 + 1):
            p1[i] = ind2.task_allocation[i]
            p2[i] = ind1.task_allocation[i]

        for i in range(size):
            if i < cxpoint1 or i > cxpoint2:
                ii = ind1.task_allocation[i]
                while self.list_dict_found(p1[cxpoint1 : cxpoint2 + 1], ii):
                    # ii = ind1.task_allocation[p1[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind1.task_allocation[
                        self.list_dict_index(ind2.task_allocation, ii)
                    ]
                p1[i] = ii

                ii = ind2.task_allocation[i]
                while self.list_dict_found(p2[cxpoint1 : cxpoint2 + 1], ii):
                    # ii = ind2.task_allocation[p2[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind2.task_allocation[
                        self.list_dict_index(ind1.task_allocation, ii)
                    ]
                p2[i] = ii

        ind1.task_allocation = p1
        ind2.task_allocation = p2
        population.append(ind1)
        population.append(ind2)

    def opt_gpu(self, population: list, ind: individual):
        ind = ind.copy()
        tasks = [
            task
            for task in ind.predict_run_seq.items()
            if task[1]['resources']['gpu'] >= 1
        ]
        tasks = sorted(tasks, key=lambda x: x[1]['total_runtime'], reverse=True)
        # 对最长运行时间的任务增加 GPU 资源
        for i in range((len(tasks) // 3)):  # 处理前1/3的任务
            index = self.list_dict_index(ind.task_allocation, tasks[i][1])
            if index:
                new_alloc = (
                    random.choice([1, 2])
                    + ind.task_allocation[index]['resources']['gpu']
                )
                max_gpu = ind.total_resources['gpu']
                if new_alloc <= max_gpu:  # 只允许在资源约束内增加
                    ind.task_allocation[index]['resources']['gpu'] = new_alloc

        # 对最短运行时间的任务减少 GPU 资源
        for i in range((len(tasks) // 3)):  # 处理后1/3的任务
            index = self.list_dict_index(ind.task_allocation, tasks[-i][1])
            if index:
                new_alloc = ind.task_allocation[index]['resources']['gpu'] - 1
                if new_alloc >= 1:  # 确保至少有 1 个 GPU 资源
                    ind.task_allocation[index]['resources']['gpu'] = new_alloc

        # 将修改后的个体添加到种群中
        population.append(ind)

    def opt1(self, population: list, ind: individual):
        ind = ind.copy()
        tasks = sorted(
            ind.predict_run_seq.items(),
            key=lambda x: x[1]['total_runtime'],
            reverse=True,
        )

        # 随机选择是增加还是减少资源
        if random.random() < 0.5:  # 50%概率增加资源
            # 对运行时间最长的任务增加资源
            for i in range((len(tasks) // 3)):
                index = self.list_dict_index(ind.task_allocation, tasks[i][1])
                if index:
                    current_cpu = ind.task_allocation[index]['resources']['cpu']
                    node = ind.task_allocation[index]['resources']['node']
                    max_cpu = self.node_resources[node]['cpu']

                    # 更激进的资源增加策略
                    if current_cpu < 4:
                        increment = random.choice([2, 3, 4])
                    elif current_cpu < 8:
                        increment = random.choice([4, 6, 8])
                    else:
                        increment = random.choice([6, 8, 10])

                    new_alloc = current_cpu + increment
                    if new_alloc <= max_cpu:  # 使用节点实际的CPU上限
                        ind.task_allocation[index]['resources']['cpu'] = new_alloc

        else:  
            # 对运行时间最短的任务减少资源
            for i in range((len(tasks) // 3)):
                index = self.list_dict_index(ind.task_allocation, tasks[-(i+1)][1])
                if index:
                    current_cpu = ind.task_allocation[index]['resources']['cpu']

                    # 更谨慎的资源减少策略
                    if current_cpu > 8:
                        decrement = random.choice([4, 6])
                    elif current_cpu > 4:
                        decrement = random.choice([2, 3])
                    else:
                        decrement = 1

                    new_alloc = current_cpu - decrement
                    if new_alloc >= 1:  # 确保至少保留1个CPU
                        ind.task_allocation[index]['resources']['cpu'] = new_alloc

        population.append(ind)

    def opt2(self, population: list, ind: individual):
        ind = ind.copy()
        ## advance the latest task order
        tasks = sorted(
            ind.predict_run_seq.items(), key=lambda x: x[1]['finish_time']
        )  # asending
        task = tasks[-1][1]
        index = self.list_dict_index(ind.task_allocation, task)
        # task may in running, so we need to check if it is in the task allocation
        if index:
            new_index = random.randrange(0, index)
            element = ind.task_allocation.pop(index)
            ind.task_allocation.insert(new_index, element)

        ## delay the earliest task order
        # task = min(ind.predict_run_seq, key=lambda x:x['start_time'])
        # index = self.list_dict_index(ind.task_allocation,task)
        # # task may in running, so we need to check if it is in the task allocation
        # if index:
        #     new_index = random.randrange(index, len(ind.task_allocation))
        #     element = ind.task_allocation.pop(index)
        #     ind.task_allocation.insert(new_index, element)
        population.append(ind)

    def process_individual_opt(self, population):
        # logger.info(f"process_infividual:{ind1.individual_id}")
        for ind in population:
            self.opt1(ind)
            self.opt2(ind)

    def process_individual_mutate(self, population, ind1, ind2, crossover_rate=0):
        self.mutate_cpu(population, ind1)
        self.mutate_cpu(population, ind2)

        self.mutate_seq(population, ind1)
        self.mutate_seq(population, ind2)

        self.crossover_pmx(population, ind1, ind2)
        self.crossover_arith_ave(population, ind1, ind2)

    def check_generation_node(self, population):
        logger_buffer = {}
        for ind in population:
            for node in ind.task_allocation_node.keys():
                if len(ind.task_allocation_node[node]) == 0:
                    logger_buffer[ind.individual_id] = f"Node {node} has no task"

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
        self.write_log(f"Initial allocation: {population[0].task_allocation}")

        # 检查边界条件
        if len(population) < 1 or len(population[0].task_allocation) < 2:
            self.write_log(f"Node {node}: No population, or only one task. Skipping...")
            return (node, population[0].task_allocation)

        score = 0
        new_score = 0

        for gen_node in range(num_runs_in_node):
            population = population[:num_generations_node]
            random.shuffle(population)
            size = len(population)

            for i in range(size // 2):
                ind1 = population[i]
                ind2 = population[size - i - 1]

                if new_score == score:
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
            scores = [self.fitness(ind, all_node=False) for ind in population]
            population = [population[i] for i in np.argsort(scores)[::-1]]
            score = new_score
            new_score = population[0].score

            # 记录每一代的关键信息
            best_ind = population[0]
            self.write_log(
                f"Node {node} Generation {gen_node + 1}: "
                f"Score = {new_score:.2f}, "
                f"Best CPU alloc = {[task['resources']['cpu'] for task in best_ind.task_allocation]}"
            )

        best_ind = max(population, key=lambda ind: ind.score)
        self.write_log(
            f"Node {node} Final Result:\n"
            f"Best Score = {best_ind.score:.2f}\n"
            f"Final Allocation = {best_ind.task_allocation}\n"
            f"{'-' * 80}"
        )

        return (node, best_ind.task_allocation)

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
        self.write_log(f"\nStarting GA with {len(all_tasks)} tasks")
        self.write_log(f"Running tasks: {self.running_task_node}")
        self.write_log(f"Available resources: {self.node_resources}")

        # run no record task
        ind = self.detect_no_his_task(all_tasks)
        if len(ind.task_allocation) > 0:
            return ind.task_allocation

        self.population = self.generate_population_all(all_tasks=all_tasks, population_size=num_generations_all)
        predict_model_train_time = time.time()
        self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
        logger.info(f"Polynomial train time: {time.time() - predict_model_train_time:.2f} seconds")
        
        self.sch_data.Task_time_predictor.estimate_ga_population(
            self.population, self.sch_data.sch_task_list, all_node=True
        )
        scores = [self.fitness(ind, all_node=True) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]

        # only one task , submit directly
        if self.at.get_total_nums(all_tasks) == 1:
            logger.info(f"Only one task, submit directly")
            self.best_ind = self.population[-1]  # predifined resources at last in list
            best_allocation = []
            for key in self.best_ind.task_allocation_node.keys(): # return a list
                best_allocation.extend(self.best_ind.task_allocation_node[key])
            return best_allocation

        # logger.info(f"Generation 0: {population[0]}")
        score = self.population[0].score
        logger.info(f"initial score is {score}")
        new_score = 0
        # for gen in range(num_runs): ## total epoch
        #     # evo on each node， population size will influence the times for ga run
        for gen, a_ind in enumerate(self.population):
            self.write_log(f"\nGlobal Generation {gen + 1}")
            # self.write_log(f"Before load balance: {a_ind.task_allocation_node}")
            # self.load_balance(a_ind)
            # self.write_log(f"After load balance: {a_ind.task_allocation_node}")
            load_balance_times = 5
            num_runs = np.linspace(1, num_runs_in_node, load_balance_times).astype(int)
            for _ in range(load_balance_times):
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
                    node, task_allocation = results
                    a_ind.task_allocation_node[node] = task_allocation
                    self.write_log(f"Node {node} final allocation: {task_allocation}")
                
                # # 并行处理每个节点
                # results = pool.starmap(
                #     self.run_ga_for_node,
                #     [
                #         (node, population, num_runs_in_node, num_generations_node)
                #         for node, population in self.population_node.items()
                #     ],
                # )
                # for node, task_allocation in results:
                #     a_ind.task_allocation_node[node] = task_allocation
                #     self.write_log(f"Node {node} final allocation: {task_allocation}")

                # all node ind operation end
        # global ind operation here
        scores = [self.fitness(ind, all_node=True) for ind in self.population]
        # logger.info(f"Generation {gen}: best ind score:{self.population[0].score}")

        # best ind global
        best_ind = max(self.population, key=lambda ind: ind.score)
        self.best_ind = best_ind
        
        self.write_log("\nFinal Results:")
        self.write_log(f"scores of all individuals: {scores}")
        self.write_log(f"Best individual score: {best_ind.score}")
        self.write_log(f"Best allocation: {best_ind.task_allocation}")
        self.write_log(f"GA running time: {time.time() - start_time:.2f} seconds")
        best_allocation = []
        for key in best_ind.task_allocation_node.keys():
            best_allocation.extend(best_ind.task_allocation_node[key])

        logger.info("GA running time: %s seconds" % (time.time() - start_time))

        return best_allocation
