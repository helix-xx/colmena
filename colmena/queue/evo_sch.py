# Standard library imports
import copy
import gc
import json
import os
import random
import time
import datetime
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, ClassVar, Collection, Dict, List, Literal, Optional, Union
from numba import jit, float64, int32

from functools import lru_cache
from xml.sax.handler import all_features

import numpy as np
import psutil

# Local application imports
from colmena.models import Result
from .monitor import available_task, HistoricalData, Sch_data
from .scheduler_util import task_dtype, distribute_tasks

# Configure logging
import logging
logging.getLogger("sklearnex").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Path configuration
# def setup_path():
#     relative_path = "~/project/colmena/multisite_"
#     absolute_path = os.path.expanduser(relative_path)
#     if absolute_path not in sys.path:
#         sys.path.append(absolute_path)

# setup_path()


def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj

import time
import json
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

class GADataCollector:
    """跟踪GA运行时的数据收集器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置收集器状态"""
        self.metrics = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': [],
            'time_per_generation': [],
            'operator_metrics': defaultdict(lambda: {
                'attempts': 0,
                'improvements': 0,
                'improvement_amounts': []
            }),
            'best_makespans': [],  # 每代最佳个体的完成时间
            'population_history': [],  # 每代的种群统计信息
            'operator_history': []  # 每代的算子使用情况
        }
        self.current_best_fitness = float('-inf')
        self.gen_start_time = None
    
    def start_generation(self):
        """开始新一代的计时"""
        self.gen_start_time = time.time()
        self.gen_operator_metrics = defaultdict(lambda: {
            'attempts': 0,
            'improvements': 0,
            'improvement_amounts': []
        })
    
    def end_generation(self, generation, population, scores):
        """结束当前代的数据收集"""
        if not self.gen_start_time:
            return
            
        gen_time = time.time() - self.gen_start_time
        best_idx = np.argmax(scores)
        best_fitness = scores[best_idx]
        avg_fitness = np.mean(scores)
        diversity = self._calculate_diversity(population)
        
        # 记录基本指标
        self.metrics['generations'].append(generation)
        self.metrics['best_fitness'].append(best_fitness)
        self.metrics['avg_fitness'].append(avg_fitness)
        self.metrics['population_diversity'].append(diversity)
        self.metrics['time_per_generation'].append(gen_time)
        
        # 如果有makespan数据，记录它
        if hasattr(population[best_idx], 'completion_time'):
            best_makespan = float(np.max(population[best_idx].completion_time))
            self.metrics['best_makespans'].append(best_makespan)
        
        # 记录种群统计
        self.metrics['population_history'].append({
            'gen': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'gen_time': gen_time
        })
        
        # 记录该代算子使用情况
        self.metrics['operator_history'].append({
            'gen': generation,
            **{op: metrics['attempts'] for op, metrics in self.gen_operator_metrics.items()}
        })
        
        # 更新当前最佳适应度
        self.current_best_fitness = max(best_fitness, self.current_best_fitness)
        
        # 合并本代算子指标到总指标
        for op_name, op_metrics in self.gen_operator_metrics.items():
            self.metrics['operator_metrics'][op_name]['attempts'] += op_metrics['attempts']
            self.metrics['operator_metrics'][op_name]['improvements'] += op_metrics['improvements']
            self.metrics['operator_metrics'][op_name]['improvement_amounts'].extend(
                op_metrics['improvement_amounts']
            )
    
    def track_operator(self, operator_name, result_ind=None):
        """
        跟踪算子的应用
        
        参数:
        operator_name: 算子名称
        result_ind: 算子产生的个体（可选）
        
        返回:
        True表示该算子产生了改进
        """
        # 记录算子被使用
        self.gen_operator_metrics[operator_name]['attempts'] += 1
        
        # 如果提供了结果个体，检查是否改进
        if result_ind is not None and hasattr(result_ind, 'score'):
            fitness = result_ind.score
            
            if fitness > self.current_best_fitness:
                improvement = fitness - self.current_best_fitness
                self.gen_operator_metrics[operator_name]['improvements'] += 1
                self.gen_operator_metrics[operator_name]['improvement_amounts'].append(improvement)
                self.current_best_fitness = fitness
                return True
        
        return False
    
    def _calculate_diversity(self, population):
        """计算种群多样性"""
        if not population or len(population) < 2:
            return 0.0
        
        # 对大种群采样以提高效率
        if len(population) > 20:
            sample_pop = random.sample(population, 20)
        else:
            sample_pop = population
        
        # 计算个体间平均差异
        total_diff = 0.0
        count = 0
        
        for i in range(len(sample_pop)):
            for j in range(i+1, len(sample_pop)):
                ind1, ind2 = sample_pop[i], sample_pop[j]
                
                # CPU分配差异
                cpu_diff = np.mean(np.abs(ind1.task_array['cpu'] - ind2.task_array['cpu']))
                
                # GPU分配差异
                gpu_diff = np.mean(np.abs(ind1.task_array['gpu'] - ind2.task_array['gpu']))
                
                # 节点分配差异
                node_diff = np.mean(ind1.task_array['node'] != ind2.task_array['node'])
                
                # 综合差异
                diff = cpu_diff + gpu_diff + node_diff * 10
                total_diff += diff
                count += 1
        
        return total_diff / max(1, count)
    
    def save_to_file(self, filename=None):
        """将收集的指标保存到文件"""
        if filename is None:
            filename = f'ga_metrics_{time.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        return filename
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
            ('task_indices', 'i4', (400,))  # 假设每个节点最多100个任务
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



# class Back_Filing_Scheduler:
#     pass


# @jit(nopython=True)
# def _calculate_completion_time(
#     task_cpu,        # shape: (n,), dtype: int32
#     task_gpu,        # shape: (n,), dtype: int32
#     task_runtime,    # shape: (n,), dtype: float64
#     running_finish_times,  # shape: (m,), dtype: float64
#     running_cpus,         # shape: (m,), dtype: int32
#     running_gpus,         # shape: (m,), dtype: int32
#     resources_cpu,   # int
#     resources_gpu,   # int
#     current_time,    # float
# ):
#     """Numba优化版本的完成时间计算"""
#     # 预分配数组
#     n_tasks = len(task_cpu)
#     n_running = len(running_finish_times)
#     max_tasks = n_tasks + n_running
    
#     ongoing_times = np.zeros(max_tasks, dtype=np.float64)
#     ongoing_cpus = np.zeros(max_tasks, dtype=np.int32)
#     ongoing_gpus = np.zeros(max_tasks, dtype=np.int32)
    
#     # 初始化资源
#     avail_cpu = resources_cpu
#     avail_gpu = resources_gpu
#     task_count = 0
#     start_time = current_time
    
#     # 添加运行中任务
#     for i in range(n_running):
#         insert_pos = task_count
#         for j in range(task_count):
#             if ongoing_times[j] > running_finish_times[i]:
#                 insert_pos = j
#                 break
#         # 移动现有任务
#         for j in range(task_count, insert_pos, -1):
#             ongoing_times[j] = ongoing_times[j - 1]
#             ongoing_cpus[j] = ongoing_cpus[j - 1]
#             ongoing_gpus[j] = ongoing_gpus[j - 1]
        
#         ongoing_times[insert_pos] = running_finish_times[i]
#         ongoing_cpus[insert_pos] = running_cpus[i]
#         ongoing_gpus[insert_pos] = running_gpus[i]
#         avail_cpu -= ongoing_cpus[insert_pos]
#         avail_gpu -= ongoing_gpus[insert_pos]
#         task_count += 1
        
#     # 记录资源使用变化点
#     changes_times = np.zeros(max_tasks * 2, dtype=np.float64)
#     changes_cpu = np.zeros(max_tasks * 2, dtype=np.int32)
#     changes_gpu = np.zeros(max_tasks * 2, dtype=np.int32)
#     changes_count = 0
    
#     if task_count > 0:
#         changes_times[0] = current_time
#         changes_cpu[0] = resources_cpu - avail_cpu
#         changes_gpu[0] = resources_gpu - avail_gpu
#         changes_count += 1

#     # 处理任务数组
#     task_starts = np.zeros(n_tasks, dtype=np.float64)
#     task_ends = np.zeros(n_tasks, dtype=np.float64)
    
#     for i in range(n_tasks):
#         required_cpu = task_cpu[i]
#         required_gpu = task_gpu[i]
#         duration = task_runtime[i]
        
#         # 检查已完成任务
#         while task_count > 0 and ongoing_times[0] <= current_time:
#             avail_cpu += ongoing_cpus[0]
#             avail_gpu += ongoing_gpus[0]
            
#             # 记录资源变化
#             changes_times[changes_count] = current_time 
#             changes_cpu[changes_count] = resources_cpu - avail_cpu
#             changes_gpu[changes_count] = resources_gpu - avail_gpu
#             changes_count += 1
            
#             # 移除完成的任务
#             for j in range(task_count - 1):
#                 ongoing_times[j] = ongoing_times[j + 1]
#                 ongoing_cpus[j] = ongoing_cpus[j + 1]
#                 ongoing_gpus[j] = ongoing_gpus[j + 1]
#             task_count -= 1
            
#         # 等待资源
#         while avail_cpu < required_cpu or avail_gpu < required_gpu:
#             if task_count == 0:
#                 return -1.0, -1.0, -1.0, task_starts, task_ends
#             current_time = ongoing_times[0]
#             avail_cpu += ongoing_cpus[0]
#             avail_gpu += ongoing_gpus[0]
            
#             # 记录资源变化
#             changes_times[changes_count] = current_time
#             changes_cpu[changes_count] = resources_cpu - avail_cpu
#             changes_gpu[changes_count] = resources_gpu - avail_gpu
#             changes_count += 1
            
#             # 移除第一个任务
#             for j in range(task_count - 1):
#                 ongoing_times[j] = ongoing_times[j + 1]
#                 ongoing_cpus[j] = ongoing_cpus[j + 1]
#                 ongoing_gpus[j] = ongoing_gpus[j + 1]
#             task_count -= 1
            
#         # 分配新任务
#         finish_time = current_time + duration
        
#         # 二分查找插入位置
#         insert_pos = task_count
#         for j in range(task_count):
#             if ongoing_times[j] > finish_time:
#                 insert_pos = j
#                 break
                
#         # 移动现有任务
#         for j in range(task_count, insert_pos, -1):
#             ongoing_times[j] = ongoing_times[j - 1]
#             ongoing_cpus[j] = ongoing_cpus[j - 1]
#             ongoing_gpus[j] = ongoing_gpus[j - 1]
            
#         # 插入新任务
#         ongoing_times[insert_pos] = finish_time
#         ongoing_cpus[insert_pos] = required_cpu
#         ongoing_gpus[insert_pos] = required_gpu
#         task_count += 1
        
#         # 记录任务时间
#         task_starts[i] = current_time
#         task_ends[i] = finish_time
        
#         avail_cpu -= required_cpu
#         avail_gpu -= required_gpu
        
#         # 记录资源变化
#         changes_times[changes_count] = current_time
#         changes_cpu[changes_count] = resources_cpu - avail_cpu
#         changes_gpu[changes_count] = resources_gpu - avail_gpu
#         changes_count += 1
    
#     while task_count > 0:
#         current_time = ongoing_times[0]
            
#         avail_cpu += ongoing_cpus[0]
#         avail_gpu += ongoing_gpus[0]
        
#         # 记录资源变化
#         changes_times[changes_count] = current_time
#         changes_cpu[changes_count] = resources_cpu - avail_cpu
#         changes_gpu[changes_count] = resources_gpu - avail_gpu
#         changes_count += 1
        
#         # 移除完成的任务
#         for j in range(task_count - 1):
#             ongoing_times[j] = ongoing_times[j + 1]
#             ongoing_cpus[j] = ongoing_cpus[j + 1]
#             ongoing_gpus[j] = ongoing_gpus[j + 1]
#         task_count -= 1
        
#     # 计算资源使用面积
#     resource_area = 0.0
#     for i in range(changes_count - 1):
#         time_delta = changes_times[i + 1] - changes_times[i]
#         area = (changes_cpu[i] + changes_gpu[i]) * time_delta
#         resource_area += area
        
#     completion_time = current_time - start_time if n_tasks > 0 else 0
#     total_runtime = completion_time
    
#     return completion_time, resource_area, total_runtime, task_starts, task_ends


@jit(nopython=True)
def _generate_right_profile(times, cpu_usage, gpu_usage):
    """生成资源使用的右轮廓线"""
    n = len(times)
    if n == 0:
        # 返回三个空float64数组，与正常返回类型保持一致
        return (np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))
    
    # 从后往前遍历，保留每个时间点之后的最大值
    max_total = 0
    right_times = []
    right_usage = []
    
    for i in range(n-1, -1, -1):
        current_total = cpu_usage[i] + gpu_usage[i]
        if current_total > max_total:
            right_times.insert(0, times[i])
            right_usage.insert(0, current_total)
            max_total = current_total
    
    # 这里确保返回的第三个数组也是float64类型
    initial_resources = np.array([float(cpu_usage[0]), float(gpu_usage[0])], dtype=np.float64)
            
    return (np.array(right_times, dtype=np.float64),
            np.array(right_usage, dtype=np.float64),
            initial_resources)
    
@jit(nopython=True)
def _calculate_completion_time_with_state(
    task_cpu,        # shape: (n,), dtype: int32
    task_gpu,        # shape: (n,), dtype: int32
    task_runtime,    # shape: (n,), dtype: float64
    current_time,    # float
    avail_cpu,       # int
    avail_gpu,       # int
    task_count,      # int
    ongoing_times,   # shape: (k,), dtype: float64
    ongoing_cpus,    # shape: (k,), dtype: int32
    ongoing_gpus,    # shape: (k,), dtype: int32
    resources_cpu,   # int
    resources_gpu,   # int
):
    """使用预计算状态的完成时间计算"""
    n_tasks = len(task_cpu)
    max_tasks = n_tasks + len(ongoing_times)
    
    # 创建新的状态数组
    new_ongoing_times = np.zeros(max_tasks, dtype=np.float64)
    new_ongoing_cpus = np.zeros(max_tasks, dtype=np.int32)
    new_ongoing_gpus = np.zeros(max_tasks, dtype=np.int32)
    
    # 复制现有状态
    new_ongoing_times[:task_count] = ongoing_times
    new_ongoing_cpus[:task_count] = ongoing_cpus
    new_ongoing_gpus[:task_count] = ongoing_gpus
    
    start_time = current_time
    task_starts = np.zeros(n_tasks, dtype=np.float64)
    task_ends = np.zeros(n_tasks, dtype=np.float64)
    
    # 记录资源变化点
    changes_times = np.zeros(max_tasks * 2 + 1, dtype=np.float64)
    changes_cpu = np.zeros(max_tasks * 2 + 1, dtype=np.int32)
    changes_gpu = np.zeros(max_tasks * 2 + 1, dtype=np.int32)
    changes_count = 0
    
    # 初始化时强制记录初始状态
    if changes_count == 0:
        changes_times[0] = current_time
        changes_cpu[0] = resources_cpu - avail_cpu
        changes_gpu[0] = resources_gpu - avail_gpu
        changes_count += 1
    
    # 定义记录资源变化的函数
    def record_change(t, cpu, gpu):
        nonlocal changes_count
        if changes_count == 0 or (cpu != changes_cpu[changes_count-1] or gpu != changes_gpu[changes_count-1]):
            changes_times[changes_count] = t
            changes_cpu[changes_count] = cpu
            changes_gpu[changes_count] = gpu
            changes_count += 1
    
    # 处理新任务
    for i in range(n_tasks):
        required_cpu = task_cpu[i]
        required_gpu = task_gpu[i]
        duration = task_runtime[i]
        
        # 检查已完成任务
        while task_count > 0 and new_ongoing_times[0] <= current_time:
            avail_cpu += new_ongoing_cpus[0]
            avail_gpu += new_ongoing_gpus[0]
            
            # 记录资源变化 - 任务完成释放资源
            record_change(current_time, resources_cpu - avail_cpu, resources_gpu - avail_gpu)
            
            for j in range(task_count - 1):
                new_ongoing_times[j] = new_ongoing_times[j + 1]
                new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
                new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
            task_count -= 1
            
        # 等待资源
        while avail_cpu < required_cpu or avail_gpu < required_gpu:
            if task_count == 0:
                return -1.0, 0, task_starts, task_ends
            current_time = new_ongoing_times[0]
            avail_cpu += new_ongoing_cpus[0]
            avail_gpu += new_ongoing_gpus[0]
            
            # 记录资源变化 - 任务完成释放资源
            record_change(current_time, resources_cpu - avail_cpu, resources_gpu - avail_gpu)
            
            for j in range(task_count - 1):
                new_ongoing_times[j] = new_ongoing_times[j + 1]
                new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
                new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
            task_count -= 1
        
        # 分配新任务
        finish_time = current_time + duration
        
        # 查找插入位置
        insert_pos = task_count
        for j in range(task_count):
            if new_ongoing_times[j] > finish_time:
                insert_pos = j
                break
                
        # 移动现有任务
        for j in range(task_count, insert_pos, -1):
            new_ongoing_times[j] = new_ongoing_times[j - 1]
            new_ongoing_cpus[j] = new_ongoing_cpus[j - 1]
            new_ongoing_gpus[j] = new_ongoing_gpus[j - 1]
        
        new_ongoing_times[insert_pos] = finish_time
        new_ongoing_cpus[insert_pos] = required_cpu
        new_ongoing_gpus[insert_pos] = required_gpu
        task_count += 1
        
        task_starts[i] = current_time
        task_ends[i] = finish_time
        
        avail_cpu -= required_cpu
        avail_gpu -= required_gpu
        
        # 记录资源变化 - 分配新任务
        record_change(current_time, resources_cpu - avail_cpu, resources_gpu - avail_gpu)
    
    # 处理剩余任务完成
    while task_count > 0:
        current_time = new_ongoing_times[0]
        avail_cpu += new_ongoing_cpus[0]
        avail_gpu += new_ongoing_gpus[0]
        
        # 记录资源变化 - 任务完成释放资源
        record_change(current_time, resources_cpu - avail_cpu, resources_gpu - avail_gpu)
        
        # 移除完成的任务
        for j in range(task_count - 1):
            new_ongoing_times[j] = new_ongoing_times[j + 1]
            new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
            new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
        task_count -= 1
    
    # 在计算resource_area前添加右轮廓线生成
    right_times, right_usage, initial_resources = _generate_right_profile(
        changes_times[:changes_count],
        changes_cpu[:changes_count],
        changes_gpu[:changes_count]
    )

    # 计算右轮廓线面积
    resource_area = 0.0
    for i in range(len(right_times)-1):
        delta = right_times[i+1] - right_times[i]
        resource_area += right_usage[i] * delta

    # 添加最终释放阶段面积
    if len(right_times) > 0:
        final_time = right_times[-1]
        final_usage = right_usage[-1]
        if current_time > final_time:
            resource_area += final_usage * (current_time - final_time)
    
    # 计算空闲面积（总资源容量 * 总时间 - 使用面积） fitness 不能使用空闲面积，否则会导致ga算法去计算最长的completion time 实现更大的空闲面积
    # total_resource = resources_cpu + resources_gpu
    # total_time = current_time - start_time
    # idle_area = total_resource * total_time - resource_area
    
    completion_time = current_time - start_time if n_tasks > 0 else 0
    return completion_time, resource_area, task_starts, task_ends


@jit(nopython=True)
def _calculate_task_resource_area(task_runtime, task_cpu, task_gpu):
    """计算任务的资源使用总面积"""
    resource_area = 0.0
    for i in range(len(task_runtime)):
        duration = task_runtime[i]
        area = (task_cpu[i] + task_gpu[i]) * duration
        resource_area += area
    return resource_area

@jit(nopython=True)
def _precalculate_fixed_tasks_state(
    queued_task_cpu,     # shape: (q,), dtype: int32
    queued_task_gpu,     # shape: (q,), dtype: int32
    queued_task_runtime, # shape: (q,), dtype: float64
    running_finish_times,  # shape: (m,), dtype: float64
    running_cpus,         # shape: (m,), dtype: int32
    running_gpus,         # shape: (m,), dtype: int32
    resources_cpu,   # int
    resources_gpu,   # int
    current_time,    # float
):
    """预计算固定任务(running和queued)的状态"""
    n_queued = len(queued_task_cpu)
    n_running = len(running_finish_times)
    max_tasks = n_queued + n_running
    
    ongoing_times = np.zeros(max_tasks, dtype=np.float64)
    ongoing_cpus = np.zeros(max_tasks, dtype=np.int32)
    ongoing_gpus = np.zeros(max_tasks, dtype=np.int32)
    
    # 初始化资源和任务计数
    avail_cpu = resources_cpu
    avail_gpu = resources_gpu
    task_count = 0
    
    # 添加running任务
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
        task_count += 1
        
        avail_cpu -= running_cpus[i]
        avail_gpu -= running_gpus[i]
    
    # 处理queued任务
    queued_starts = np.zeros(n_queued, dtype=np.float64)
    queued_ends = np.zeros(n_queued, dtype=np.float64)
    
    for i in range(n_queued):
        required_cpu = queued_task_cpu[i]
        required_gpu = queued_task_gpu[i]
        duration = queued_task_runtime[i]
        
        # 检查已完成任务
        while task_count > 0 and ongoing_times[0] <= current_time:
            avail_cpu += ongoing_cpus[0]
            avail_gpu += ongoing_gpus[0]
            
            for j in range(task_count - 1):
                ongoing_times[j] = ongoing_times[j + 1]
                ongoing_cpus[j] = ongoing_cpus[j + 1]
                ongoing_gpus[j] = ongoing_gpus[j + 1]
            task_count -= 1
            
        # 等待资源
        while avail_cpu < required_cpu or avail_gpu < required_gpu:
            if task_count == 0:
                return None  # 资源分配失败
            current_time = ongoing_times[0]
            avail_cpu += ongoing_cpus[0]
            avail_gpu += ongoing_gpus[0]
            
            for j in range(task_count - 1):
                ongoing_times[j] = ongoing_times[j + 1]
                ongoing_cpus[j] = ongoing_cpus[j + 1]
                ongoing_gpus[j] = ongoing_gpus[j + 1]
            task_count -= 1
        
        # 分配新任务
        finish_time = current_time + duration
        
        # 查找插入位置
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
        
        ongoing_times[insert_pos] = finish_time
        ongoing_cpus[insert_pos] = required_cpu
        ongoing_gpus[insert_pos] = required_gpu
        task_count += 1
        
        queued_starts[i] = current_time
        queued_ends[i] = finish_time
        
        avail_cpu -= required_cpu
        avail_gpu -= required_gpu
    
    return current_time, avail_cpu, avail_gpu, task_count, ongoing_times[:task_count], \
           ongoing_cpus[:task_count], ongoing_gpus[:task_count], queued_starts, queued_ends

# @jit(nopython=True)
# def _calculate_completion_time_with_state(
#     task_cpu,        # shape: (n,), dtype: int32
#     task_gpu,        # shape: (n,), dtype: int32
#     task_runtime,    # shape: (n,), dtype: float64
#     current_time,    # float
#     avail_cpu,       # int
#     avail_gpu,       # int
#     task_count,      # int
#     ongoing_times,   # shape: (k,), dtype: float64
#     ongoing_cpus,    # shape: (k,), dtype: int32
#     ongoing_gpus,    # shape: (k,), dtype: int32
#     resources_cpu,   # int
#     resources_gpu,   # int
# ):
#     """使用预计算状态的完成时间计算"""
#     n_tasks = len(task_cpu)
#     max_tasks = n_tasks + len(ongoing_times)
    
#     # 创建新的状态数组
#     new_ongoing_times = np.zeros(max_tasks, dtype=np.float64)
#     new_ongoing_cpus = np.zeros(max_tasks, dtype=np.int32)
#     new_ongoing_gpus = np.zeros(max_tasks, dtype=np.int32)
    
#     # 复制现有状态
#     new_ongoing_times[:task_count] = ongoing_times
#     new_ongoing_cpus[:task_count] = ongoing_cpus
#     new_ongoing_gpus[:task_count] = ongoing_gpus
    
#     start_time = current_time
#     task_starts = np.zeros(n_tasks, dtype=np.float64)
#     task_ends = np.zeros(n_tasks, dtype=np.float64)
    
#     # 处理新任务
#     for i in range(n_tasks):
#         required_cpu = task_cpu[i]
#         required_gpu = task_gpu[i]
#         duration = task_runtime[i]
        
#         # 检查已完成任务
#         while task_count > 0 and new_ongoing_times[0] <= current_time:
#             avail_cpu += new_ongoing_cpus[0]
#             avail_gpu += new_ongoing_gpus[0]
            
#             for j in range(task_count - 1):
#                 new_ongoing_times[j] = new_ongoing_times[j + 1]
#                 new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
#                 new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
#             task_count -= 1
            
#         # 等待资源
#         while avail_cpu < required_cpu or avail_gpu < required_gpu:
#             if task_count == 0:
#                 return -1.0, 0, task_starts, task_ends
#             current_time = new_ongoing_times[0]
#             avail_cpu += new_ongoing_cpus[0]
#             avail_gpu += new_ongoing_gpus[0]
            
#             for j in range(task_count - 1):
#                 new_ongoing_times[j] = new_ongoing_times[j + 1]
#                 new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
#                 new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
#             task_count -= 1
        
#         # 分配新任务
#         finish_time = current_time + duration
        
#         # 查找插入位置
#         insert_pos = task_count
#         for j in range(task_count):
#             if new_ongoing_times[j] > finish_time:
#                 insert_pos = j
#                 break
                
#         # 移动现有任务
#         for j in range(task_count, insert_pos, -1):
#             new_ongoing_times[j] = new_ongoing_times[j - 1]
#             new_ongoing_cpus[j] = new_ongoing_cpus[j - 1]
#             new_ongoing_gpus[j] = new_ongoing_gpus[j - 1]
        
#         new_ongoing_times[insert_pos] = finish_time
#         new_ongoing_cpus[insert_pos] = required_cpu
#         new_ongoing_gpus[insert_pos] = required_gpu
#         task_count += 1
        
#         task_starts[i] = current_time
#         task_ends[i] = finish_time
        
#         avail_cpu -= required_cpu
#         avail_gpu -= required_gpu
        
#     # 计算空闲资源面积
#     resources_released_weighted = 0
#     resources_released_weighted += (avail_cpu + avail_gpu) * current_time
#     while task_count > 0:
#         current_time = new_ongoing_times[0]
#         avail_cpu += new_ongoing_cpus[0]
#         avail_gpu += new_ongoing_gpus[0]
#         resources_released_weighted += (new_ongoing_cpus[0] + new_ongoing_gpus[0]) * current_time
        
#         # 移除完成的任务
#         for j in range(task_count - 1):
#             new_ongoing_times[j] = new_ongoing_times[j + 1]
#             new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
#             new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
#         task_count -= 1
    
#     # resources_released_weighted = current_time * (avail_cpu + avail_gpu) - resources_released_weighted
    
#     completion_time = current_time - start_time if n_tasks > 0 else 0
#     return completion_time, resources_released_weighted, task_starts, task_ends


# 封装两个 numba 函数，添加缓存
@lru_cache(maxsize=512)
def cached_calculate_completion_time_with_state(
    task_cpu_tuple, task_gpu_tuple, task_runtime_tuple,
    current_time, avail_cpu, avail_gpu,
    task_count, ongoing_times_tuple, ongoing_cpus_tuple, ongoing_gpus_tuple,
    resources_cpu, resources_gpu
):
    # 将 tuple 转为 numpy 数组
    task_cpu = np.array(task_cpu_tuple, dtype=np.int32)
    task_gpu = np.array(task_gpu_tuple, dtype=np.int32)
    task_runtime = np.array(task_runtime_tuple, dtype=np.float64)
    ongoing_times = np.array(ongoing_times_tuple, dtype=np.float64)
    ongoing_cpus = np.array(ongoing_cpus_tuple, dtype=np.int32)
    ongoing_gpus = np.array(ongoing_gpus_tuple, dtype=np.int32)

    # 调用原始 numba 函数
    return _calculate_completion_time_with_state(
        task_cpu, task_gpu, task_runtime,
        current_time, avail_cpu, avail_gpu,
        task_count, ongoing_times, ongoing_cpus, ongoing_gpus,
        resources_cpu, resources_gpu
    )

@lru_cache(maxsize=512)
def cached_calculate_task_resource_area(
    task_runtime_tuple, task_cpu_tuple, task_gpu_tuple
):
    # 将 tuple 转为 numpy 数组
    task_runtime = np.array(task_runtime_tuple, dtype=np.float64)
    task_cpu = np.array(task_cpu_tuple, dtype=np.int32)
    task_gpu = np.array(task_gpu_tuple, dtype=np.int32)

    # 调用原始 numba 函数
    return _calculate_task_resource_area(task_runtime, task_cpu, task_gpu)

def precalculate_fixed_state(sch_data:Sch_data, running_tasks_all, queued_tasks_all, scheduler_time=time.time()):
    """预计算固定任务状态"""
    sch_data.fixed_state = {}
    for node in sch_data.available_resources.keys():
        running_tasks = running_tasks_all[node]
        queued_tasks = queued_tasks_all[queued_tasks_all['node'] == node]
        
        # 转换running_tasks为数组
        if running_tasks is not None and len(running_tasks)>0:
            running_finish_times = np.array([task['finish_time'] for task in running_tasks], dtype=np.float64)
            running_cpus = np.array([task['cpu'] for task in running_tasks], dtype=np.int32)
            running_gpus = np.array([task['gpu'] for task in running_tasks], dtype=np.int32)
        else:
            running_finish_times = np.array([], dtype=np.float64)
            running_cpus = np.array([], dtype=np.int32)
            running_gpus = np.array([], dtype=np.int32)
            
        # 转换queued_tasks为数组
        if queued_tasks is not None and len(queued_tasks)>0:
            queued_task_cpu = np.array([task['cpu'] for task in queued_tasks], dtype=np.int32)
            queued_task_gpu = np.array([task['gpu'] for task in queued_tasks], dtype=np.int32)
            queued_task_runtime = np.array([task['total_runtime'] for task in queued_tasks], dtype=np.float64)
        else:
            queued_task_cpu = np.array([], dtype=np.int32)
            queued_task_gpu = np.array([], dtype=np.int32)
            queued_task_runtime = np.array([], dtype=np.float64)
            
        # 预计算状态
        sch_data.fixed_state[node] = _precalculate_fixed_tasks_state(
            queued_task_cpu,
            queued_task_gpu,
            queued_task_runtime,
            running_finish_times,
            running_cpus,
            running_gpus,
            sch_data.available_resources[node]['cpu'],
            sch_data.available_resources[node]['gpu'],
            scheduler_time
        )
        
    if sch_data.fixed_state is None:
        raise ValueError("Fixed tasks resource allocation failed")


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
        
        self.current_time = (
            0  # current running time for compute  while trigger evo_scheduler
        )

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

    def write_log(self, *messages):
        with open(self.log_path, 'a') as f:
            for message in messages:
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
            for task in  self.sch_data.running_task_node[node]:
                if task['task_id'] == result_obj.task_id:
                    self.sch_data.running_task_node[node].remove(task)

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

    def calc_used_area(self, tasks: np.ndarray) -> tuple[float, float]:
        """计算任务的CPU和GPU使用的面积(使用时间*使用数量)
        
        Args:
            tasks: 任务数组
        
        Returns:
            tuple: (CPU总时间, GPU总时间)
        """
        total_cpu_time = np.sum(tasks['cpu'] * tasks['total_runtime'])
        total_gpu_time = np.sum(tasks['gpu'] * tasks['total_runtime'])
        return total_cpu_time, total_gpu_time

    def calc_utilization(self, ind: individual) -> tuple[dict, dict, dict]:
        """计算各节点的资源利用情况, total_area指任务使用资源的总面积, released_area指到最后任务执行完开始释放资源时占用的总面积(包括中间的idle部分)
        
        Args:
            ind: individual对象
        
        Returns:
            tuple: (CPU时间字典, GPU时间字典, 完成时间字典)
        """
        total_cpu_area = defaultdict(float)
        total_gpu_area = defaultdict(float)
        completion_time = defaultdict(float)
        total_runtime = defaultdict(float)
        released_weighted = defaultdict(float)
        
        for node in self.node_resources.keys():
            node_mask = ind.task_array['node'] == node
            node_tasks = ind.task_array[node_mask]
            
            total_cpu_area[node], total_gpu_area[node] = self.calc_used_area(node_tasks)
            
            completion_time[node], released_weighted[node], total_runtime[node] = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
                node_tasks,
                ind
            )

        return total_cpu_area, total_gpu_area, completion_time, total_runtime


    # def load_balance(self, ind: individual) -> None:
    #     """平衡各节点的负载（改进版）"""
    #     if not isinstance(ind, individual):
    #         raise ValueError("load_balance input is not individual")
        
    #     # 初始化各节点资源使用状态
    #     total_cpu_time, total_gpu_time, completion_time, end_time = self.calc_utilization(ind)
    #     makespan = max(end_time.values())
        
    #     improved = True
    #     while improved:
    #         improved = False
    #         current_makespan = max(end_time.values())
            
    #         # 按完成时间降序排列节点（最忙的节点优先处理）
    #         sorted_nodes = sorted(end_time.items(), key=lambda x: -x[1])
            
    #         # 遍历所有高负载节点（允许扩展检查范围）
    #         for max_node, _ in sorted_nodes:
    #             # 如果当前节点已经不是最忙的则跳过
    #             if end_time[max_node] < current_makespan:
    #                 continue
                
    #             # 获取该节点所有任务（按执行时间降序排列）
    #             node_mask = ind.task_array['node'] == max_node
    #             tasks = ind.task_array[node_mask]
    #             sorted_tasks = sorted(tasks, key=lambda x: -x['total_runtime'])
                
    #             # 遍历所有可能迁移的任务
    #             for task in sorted_tasks:
    #                 best_target = None
    #                 best_reduction = 0
    #                 original_time = end_time[max_node]
                    
    #                 # 尝试迁移到所有其他节点
    #                 for target_node in end_time.keys():
    #                     if target_node == max_node:
    #                         continue
                        
    #                     # 检查目标节点资源是否满足
    #                     target_res = self.node_resources[target_node]
    #                     if (task['cpu'] > target_res['cpu'] or 
    #                         task['gpu'] > target_res['gpu']):
    #                         continue
                        
    #                     # 模拟迁移计算新时间
    #                     new_max_time = original_time - task['total_runtime'] * (
    #                         task['cpu'] / self.resources_evo[max_node]['cpu'] + 
    #                         task['gpu'] / self.resources_evo[max_node]['gpu'])
                        
    #                     # 估算目标节点新时间（考虑资源争用）
    #                     target_task_time = task['total_runtime'] * (
    #                         task['cpu'] / self.resources_evo[target_node]['cpu'] + 
    #                         task['gpu'] / self.resources_evo[target_node]['gpu'])
    #                     new_target_time = end_time[target_node] + target_task_time
                        
    #                     # 计算新makespan
    #                     potential_makespan = max(new_max_time, new_target_time)
    #                     if potential_makespan < current_makespan:
    #                         reduction = current_makespan - potential_makespan
    #                         if reduction > best_reduction:
    #                             best_reduction = reduction
    #                             best_target = target_node
                    
    #                 # 执行最优迁移
    #                 if best_target is not None:
    #                     # 更新任务分配
    #                     task_idx = ind.get_task_index(task['task_id'])
    #                     ind.task_array[task_idx]['node'] = best_target
                        
    #                     # 更新资源统计
    #                     task_cpu = task['cpu'] * task['total_runtime']
    #                     task_gpu = task['gpu'] * task['total_runtime']
    #                     total_cpu_time[max_node] -= task_cpu
    #                     total_gpu_time[max_node] -= task_gpu
    #                     total_cpu_time[best_target] += task_cpu
    #                     total_gpu_time[best_target] += task_gpu
                        
    #                     # 重新计算完成时间
    #                     for node in [max_node, best_target]:
    #                         node_mask = ind.task_array['node'] == node
    #                         completion_time[node], _, end_time[node] = (
    #                             self.calculate_completion_time_record_with_running_task(
    #                                 self.node_resources[node],
    #                                 ind.task_array[node_mask],
    #                                 ind
    #                             )
    #                         )
                        
    #                     improved = True
    #                     break  # 每节点每次只迁移一个任务避免震荡
                    
    #                 if improved:
    #                     break  # 如果已经改进，进入下一轮平衡
    #             if improved:
    #                 break  # 如果已经改进，进入下一轮平衡
                        
    #     ind.init_node_array()
    
    
    def load_balance(self, ind: individual) -> None:
        """平衡各节点的负载
        
        Args:
            ind: individual对象
        """
        if not isinstance(ind, individual):
            raise ValueError("load_balance input is not individual")

        total_cpu_time, total_gpu_time, completion_time, end_time = self.calc_utilization(ind)
        diff_cur = max(end_time.values()) - min(end_time.values())
        diff_pre = max(end_time.values())
        max_pre = max(end_time.values())
        max_node = max(end_time, key=end_time.get)
        min_node = min(end_time, key=end_time.get)

        while diff_cur < diff_pre:
            max_node = max(end_time, key=end_time.get)
            min_node = min(end_time, key=end_time.get)

            best_task_idx = None
            best_diff = float('inf')
            
            # 获取节点资源，避免任务到无法运行的节点上
            min_node_resources = self.node_resources[min_node]

            # 获取最大负载节点的任务
            max_node_mask = ind.task_array['node'] == max_node
            max_node_tasks = ind.task_array[max_node_mask]
            
            # 遍历最大负载节点的每个任务
            for i, task in enumerate(max_node_tasks):
                if task['cpu'] > min_node_resources['cpu'] or task['gpu'] > min_node_resources['gpu']:
                    continue
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
                
                completion_time[max_node], _, end_time[max_node] = self.calculate_completion_time_record_with_running_task(
                    self.node_resources[max_node],
                    ind.task_array[max_node_mask],
                    ind
                )
                
                completion_time[min_node], _, end_time[min_node] = self.calculate_completion_time_record_with_running_task(
                    self.node_resources[min_node],
                    ind.task_array[min_node_mask],
                    ind
                )

                diff_pre = diff_cur
                max_now = max(end_time.values())
                diff_cur = max(end_time.values()) - min(end_time.values())
                
                if max_now > max_pre:
                    break
                else:
                    max_pre = max_now
            else:
                break
            
        ind.init_node_array()
        
    def distributed_individual_tasks(self, 
                            population: list[individual],
                            ind: individual
                            ):
        """
        基于资源面积的负载均衡优化（与原distribute_tasks一致）
        """
        balanced_ind = ind.copy()
        node_loads = defaultdict(lambda: {'cpu_area':0.0, 'gpu_area':0.0})
        
        # 初始化负载计算
        for task in balanced_ind.task_array:
            node = task['node']
            runtime = self.sch_data.Task_time_predictor.get_runtime(
                task['cpu'], task['gpu'], 
                self.sch_data.sch_task_list[task['task_id']]['message_sizes.inputs'],
                task['name']
            )
            node_loads[node]['cpu_area'] += task['cpu'] * runtime
            node_loads[node]['gpu_area'] += task['gpu'] * runtime

        # 排序策略与原函数一致
        sorted_tasks = sorted(
            balanced_ind.task_array,
            key=lambda t: (t['gpu']*t['total_runtime'], t['cpu']*t['total_runtime']),
            reverse=True
        )

        # 重分配主逻辑
        for task in sorted_tasks:
            current_node = task['node']
            candidate_nodes = [
                node for node, res in self.node_resources.items()
                if res['cpu'] >= task['cpu'] and res['gpu'] >= task['gpu']
            ]
            
            if not candidate_nodes:
                continue
                
            # 选择最小化最大利用率的节点
            best_node = min(
                candidate_nodes,
                key=lambda n: max(
                    (node_loads[n]['cpu_area'] + task['cpu']*task['total_runtime']) / self.node_resources[n]['cpu'],
                    (node_loads[n]['gpu_area'] + task['gpu']*task['total_runtime']) / self.node_resources[n]['gpu'] if self.node_resources[n]['gpu']>0 else 0
                )
            )
            
            if best_node != current_node:
                # 更新负载记录
                runtime = task['total_runtime']
                node_loads[current_node]['cpu_area'] -= task['cpu'] * runtime
                node_loads[current_node]['gpu_area'] -= task['gpu'] * runtime
                node_loads[best_node]['cpu_area'] += task['cpu'] * runtime
                node_loads[best_node]['gpu_area'] += task['gpu'] * runtime
                
                # 更新任务分配
                task_idx = balanced_ind.get_task_index(task['task_id'])
                balanced_ind.task_array[task_idx]['node'] = best_node

        # 保持个体结构完整性
        balanced_ind.update_task_id_index()
        balanced_ind.init_node_array()
        
        population.append(balanced_ind)


# 修改 calculate_completion_time_record_with_running_task 使用缓存函数
    def calculate_completion_time_record_with_running_task(self, resources, task_array, ind):
        """计算完成时间并更新任务时间记录"""
        if len(task_array) == 0:
            return 0.0, 0, 0.0

        node = task_array[0]['node']
        if self.sch_data.fixed_state[node] is None:
            raise ValueError("Fixed state not calculated")

        # 提取简单数组
        task_cpu = np.array([task['cpu'] for task in task_array], dtype=np.int32)
        task_gpu = np.array([task['gpu'] for task in task_array], dtype=np.int32)
        task_runtime = np.array([task['total_runtime'] for task in task_array], dtype=np.float64)

        # 转换为 tuple 以支持缓存
        task_cpu_tuple = tuple(task_cpu)
        task_gpu_tuple = tuple(task_gpu)
        task_runtime_tuple = tuple(task_runtime)
        ongoing_times_tuple = tuple(self.sch_data.fixed_state[node][4])
        ongoing_cpus_tuple = tuple(self.sch_data.fixed_state[node][5])
        ongoing_gpus_tuple = tuple(self.sch_data.fixed_state[node][6])

        # 使用缓存的 _calculate_completion_time_with_state
        try:
            completion_time, resources_released_weighted, starts, ends = cached_calculate_completion_time_with_state(
                task_cpu_tuple,
                task_gpu_tuple,
                task_runtime_tuple,
                self.sch_data.fixed_state[node][0],  # current_time
                self.sch_data.fixed_state[node][1],  # avail_cpu
                self.sch_data.fixed_state[node][2],  # avail_gpu
                self.sch_data.fixed_state[node][3],  # task_count
                ongoing_times_tuple,
                ongoing_cpus_tuple,
                ongoing_gpus_tuple,
                resources['cpu'],
                resources['gpu']
            )
        except Exception as e:
            self.write_log("_calculate_completion_time_with_state error")
            self.write_log(f"Error occurred during resource allocation: {e}")
            self.write_log(self.sch_data.fixed_state[node], task_cpu, task_gpu, task_runtime)
            raise e

        if completion_time <= 0:
            self.write_log("_calculate_completion_time_with_state error")
            self.write_log(task_array)
            self.write_log(self.sch_data.fixed_state)
            raise ValueError("Resource allocation failed")

        # 使用缓存的 _calculate_task_resource_area
        resource_area = cached_calculate_task_resource_area(
            task_runtime_tuple, task_cpu_tuple, task_gpu_tuple
        )

        # 更新 individual 中的任务时间
        for i, task in enumerate(task_array):
            idx = ind._task_id_index[task['task_id']]
            ind.task_array[idx]['start_time'] = starts[i]
            ind.task_array[idx]['finish_time'] = ends[i]

        total_runtime = np.max(ends)

        return completion_time, resources_released_weighted, total_runtime

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
        
        resources_area_weight = 0
        for i, node in enumerate(unique_nodes):
            node_mask = ind.task_array['node'] == node
            node_tasks = ind.task_array[node_mask]
            # 添加空任务检查
            if len(node_tasks) == 0:
                completion_times[i] = 0
                resource_areas[i] = 0
                total_runtimes[i] = 0
                continue
            
            completion_time, resource_area, total_runtime = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
                node_tasks,
                ind  # 传入individual对象
            )
            
            completion_times[i] = completion_time
            resource_areas[i] = resource_area
            total_runtimes[i] = total_runtime
            
            node_total_resources = sum(self.node_resources[node].values())
            resources_area_weight += (resource_area / node_total_resources)
        
        # 存储调度指标
        ind.completion_time = completion_times
        ind.resource_area = resource_areas
        ind.total_runtime = total_runtimes # last task finish time
        
        # 计算适应度分数
        ind.score = -np.max(ind.completion_time) - 0.1*resources_area_weight
        # ind.score = -ind.completion_time
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
        for _ in range(max(population_size-2,1)):
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
    
    def generate_balanced_population(self, all_tasks, population_size: int):
        """使用MRSA的负载均衡策略生成初始化种群"""
        def create_individual_from_distribution(distributed_tasks):
            """从分布结果创建individual对象"""
            ind = individual(
                tasks_nums=len(distributed_tasks),
                total_resources=self.node_resources
            )
            
            # 转换任务格式并填充运行时
            task_list = []
            for task in distributed_tasks:
                # 获取任务元数据
                task_id = task['task_id']
                task_name = task['name']
                
                task_list.append((
                    task_name, task_id,
                    task['cpu'], task['gpu'],
                    task['node'], 0.0,
                    0.0, 0.0  # start_time和finish_time初始化为0
                ))
            
            # 填充到numpy结构化数组
            ind.task_array = np.array(
                task_list,
                dtype=ind.dtype
            )
            ind.update_task_id_index()
            return ind

        population = []
        
        # 使用改进的distribute_tasks进行任务分配
        distributed_tasks = distribute_tasks(
            tasks=all_tasks,
            nodes=self.node_resources,
            sch_task_lists=self.sch_data.sch_task_list,
            Task_time_predictor=self.sch_data.Task_time_predictor
        )
        balanced_ind = create_individual_from_distribution(distributed_tasks)
        population.append(balanced_ind)
        # 生成基于负载均衡的个体
        for _ in range(population_size//2):
            
            # 创建个体并添加到种群
            ind = balanced_ind.copy()
            ind.task_array_shuffled()  # shuffle task array
            ind.init_node_array()
            population.append(ind)
        
        # 添加随机个体保持多样性
        population += self.generate_population_all(all_tasks, max(population_size//2, 2))
        
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
                
                node_cpu = self.node_resources[new_ind.task_array[idx1]['node']]['cpu']
                new_cpu = (ind1.task_array[idx1]['cpu'] + ind2.task_array[idx2]['cpu']) // 2
                new_ind.task_array[idx1]['cpu'] = min(new_cpu, node_cpu)  
                
                node_gpu = self.node_resources[new_ind.task_array[idx1]['node']]['gpu']
                new_gpu = (ind1.task_array[idx1]['gpu'] + ind2.task_array[idx2]['gpu']) // 2
                new_ind.task_array[idx1]['gpu'] = min(new_gpu, node_gpu)  
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
            if top_count > 0:
                top_indices = sorted_indices[:top_count]
                
                # 获取对应节点的GPU限制
                selected_nodes = gpu_tasks[top_indices]['node']
                node_limits = np.array([self.node_resources[node]['gpu'] for node in selected_nodes])
                
                increments = np.random.choice([1, 2], size=top_count)
                new_gpu = np.clip(gpu_tasks[top_indices]['gpu'] + increments, 
                                1, node_limits)
                
                task_array[gpu_mask][top_indices]['gpu'] = new_gpu
            
            # 批量调整后1/3
            if top_count > 0:
                bottom_indices = sorted_indices[-top_count:]
                
                # 获取对应节点的GPU限制
                selected_nodes = gpu_tasks[bottom_indices]['node']
                node_limits = np.array([self.node_resources[node]['gpu'] for node in selected_nodes])
                
                decrements = np.random.choice([1], size=top_count)
                new_gpu = np.clip(gpu_tasks[bottom_indices]['gpu'] - decrements,
                                1, node_limits)
                
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
        
    def validate_resource_for_task(task, node, resources):
        """验证任务资源是否符合节点容量"""
        return (task['cpu'] <= resources[node]['cpu'] and 
                task['gpu'] <= resources[node]['gpu'])
        
    # 需要测试是否有效
    def node_migration_mutation(self, population: list, ind: individual):
        """节点迁移变异 - 尝试将任务迁移到其他合适的节点"""
        new_ind = ind.copy()
        task_array = new_ind.task_array
        
        # 随机选择一个节点和该节点上的任务
        nodes_with_tasks = np.unique(task_array['node'])  # unique操作可能耗时多
        if len(nodes_with_tasks) < 2:  # 需要至少两个节点才能迁移
            return
            
        source_node = random.choice(nodes_with_tasks)
        source_mask = task_array['node'] == source_node
        source_tasks = task_array[source_mask]
        
        if len(source_tasks) == 0:
            return
            
        # 随机选择一个任务
        task_idx = random.choice(range(len(source_tasks)))
        task = source_tasks[task_idx]
        global_idx = new_ind.get_task_index(task['task_id'])
        
        # 选择其他可能的目标节点
        other_nodes = [node for node in self.node_resources.keys() if node != source_node]
        random.shuffle(other_nodes)
        
        # 尝试迁移到其他节点
        for target_node in other_nodes:
            # 检查目标节点是否有足够资源
            if self.validate_resource_for_task(task, target_node, self.node_resources):
                
                # 迁移任务
                task_array[global_idx]['node'] = target_node
                new_ind.update_task_id_index()
                
                # 验证没有重复task_id
                task_ids = [t['task_id'] for t in task_array]
                assert len(set(task_ids)) == len(task_ids), f"Duplicate task_ids found after migration: {task_ids}"
                
                population.append(new_ind)
                return True
                
        return False  # 没有找到合适的目标节点

    def process_individual_opt(self, population):
        # logger.info(f"process_infividual:{ind1.individual_id}")
        for ind in population:
            self.opt1(ind)
            self.opt2(ind)


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

        # 检查边界条件
        task_nums = len(population[0].task_array)
        if len(population) < 1:
            self.write_log(f"Node {node}: No population. Return")
            return (node, population[0])
        if task_nums < 1:
            self.write_log(f"No Task. Return")
            return (node, None)
        score = 0
        new_score = 0

        for gen_node in range(num_runs_in_node):
            population = population[:num_generations_node]
            random.shuffle(population)
            size = len(population)
            for i in range(size // 2):
                ind1 = population[i]
                ind2 = population[size - i - 1]
                if task_nums > 1:
                    self.mutate_seq(population, ind1)
                    self.mutate_seq(population, ind2)
                    self.crossover_pmx(population, ind1, ind2)
                    self.opt2(population, ind1)
                    self.opt2(population, ind2)
                    
                self.mutate_resources(population, ind1)
                self.mutate_resources(population, ind2)
                self.crossover_arith_ave(population, ind1, ind2)

                self.opt1(population, ind1)
                self.opt1(population, ind2)
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
        self.write_log(f"Running tasks: {self.sch_data.running_task_node}")
        self.write_log(f"Available resources: {self.node_resources}")
        
        # fill features from new task. move to scheduler_core run_sch
        # self.sch_data.Task_time_predictor.fill_features_from_new_task(self.node_resources, self.sch_data.sch_task_list)
        # self.sch_data.Task_time_predictor.fill_runtime_records_with_predictor()
        # self.write_log(f"Predictor filled with new task features, consuming time: {time.time() - start_time:.2f} seconds")
        
        # run no record task
        ind = self.detect_no_his_task(all_tasks)
        if ind is not None and len(ind.task_array) > 0:
            return ind.task_array

        self.population = self.generate_population_all(all_tasks=all_tasks, population_size=num_generations_all)
        
        predict_model_train_time = time.time()
        self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
        logger.info(f"Predictor train time: {time.time() - predict_model_train_time:.2f} seconds")
        
        self.sch_data.Task_time_predictor.estimate_ga_population(
            self.population, self.sch_data.sch_task_list, all_node=True
        )
        scores = [self.fitness(ind) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]

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
                    self.write_log(f"\nNode {node} GA start")
                    self.write_log(f"Initial allocation: {population[0].task_array}")
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
                
                # 并行处理每个节点
                # results = pool.starmap(
                #     self.run_ga_for_node,
                #     [
                #         (node, population, num_runs_in_node, num_generations_node)
                #         for node, population in self.population_node.items()
                #     ],
                # )
                # for node, best_ind in results:
                #     self.write_log(f"Node {node} final allocation: {best_ind.task_array}")
                #     self.update_node_tasks(a_ind, best_ind, node)

                # all node ind operation end
        # global ind operation here
        scores = [self.fitness(ind) for ind in self.population]
        # logger.info(f"Generation {gen}: best ind score:{self.population[0].score}")

        # best ind global
        best_ind = max(self.population, key=lambda ind: ind.score)
        self.sch_data.best_ind = best_ind
        # best_allocation = best_ind.task_allocation
        best_allocation = best_ind.task_array
        self.write_log("\nFinal Results:")
        self.write_log(f"scores of all individuals: {scores}")
        self.write_log(f"Best individual score: {best_ind.score}")
        self.write_log(f"Best allocation: {best_allocation}")
        self.write_log(f"GA running time: {time.time() - start_time:.2f} seconds")


        logger.info("GA running time: %s seconds" % (time.time() - start_time))
        # self.at.move_allocation_to_scheduled(all_tasks, best_allocation) # should consider lock
        
        ## necessary clean
        # self.sch_data.fixed_state = {}
        return best_allocation
    
    def run_ga_v2(
        self,
        all_tasks:list[dict[str, int]],
        num_runs: int = 50,
        # num_runs_in_node: int = 5,  # 减少节点内迭代次数
        num_generations_all: int = 50,  # 增加全局种群大小,精英总群大小
        # num_generations_node: int = 20,  # 减少节点内种群大小
        pool = None,
    )->list:
        start_time = time.time()
        task_nums = self.at.get_task_nums(all_tasks)
        self.write_log(f"\nStarting GA with {task_nums} tasks, tasks list: {all_tasks}")
        self.write_log(f"Running tasks: {self.sch_data.running_task_node}")
        self.write_log(f"Available resources: {self.node_resources}")
        
        # 检测无历史任务
        ind = self.detect_no_his_task(all_tasks)
        if ind is not None and len(ind.task_array) > 0:
            return ind.task_array

        # 生成初始全局种群
        self.population = self.generate_balanced_population(all_tasks=all_tasks, population_size=num_generations_all)
        
        # 预估任务运行时间
        self.sch_data.Task_time_predictor.estimate_ga_population(
            self.population, self.sch_data.sch_task_list, all_node=True
        )
        
        # 计算初始适应度并排序
        scores = [self.fitness(ind) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]
        
        score = self.population[0].score
        logger.info(f"Initial score: {score}")
        self.write_log(f"Initial score: {score}")
        new_score = 0
        
        # 全局GA主循环
        for global_gen in range(num_runs):
            self.write_log(f"\nGlobal Generation {global_gen + 1}")
            
            # 全局种群变异与交叉
            offspring = []
            
            # 精英保留, 后续合并了父代和子代，已经保留
            # elite_size = max(1, len(self.population) // 5)
            # offspring.extend([ind.copy() for ind in self.population[:elite_size]])
            
            # 基本操作次数
            n_operations = len(self.population) * 2
            
            # 交叉
            for _ in range(n_operations):
                if len(self.population) > 1:
                    i, j = random.sample(range(len(self.population)), 2)
                    ind1, ind2 = self.population[i], self.population[j]
                    
                    if random.random() < 0.3:  # 交叉概率
                        if random.random() < 0.5:
                            self.crossover_pmx(offspring, ind1, ind2)
                        else:
                            self.crossover_arith_ave(offspring, ind1, ind2)
            
            # 变异
            for _ in range(n_operations):
                ind = random.choice(self.population)
                
                # 顺序变异
                if random.random() < 0.3 and len(ind.task_array)>=2:
                    if random.random() < 0.5:
                        self.mutate_seq(offspring, ind)
                    else:
                        shuffled_ind = ind.copy()
                        shuffled_ind.task_array_shuffled()  # shuffle task array
                        shuffled_ind.init_node_array()
                        offspring.append(shuffled_ind)
                
                # 资源变异
                if random.random() < 0.3:
                    self.mutate_resources(offspring, ind)
                
                # 节点迁移变异
                # if random.random() < 0.3:
                #     self.node_migration_mutation(offspring, ind)
            
            # 优化操作
            for _ in range(n_operations):
                ind = random.choice(self.population)
                
                # 各种优化器
                if random.random() < 0.7:
                    self.opt1(offspring, ind)
                
                if random.random() < 0.7:
                    self.opt2(offspring, ind)
                
                if random.random() < 0.7:
                    self.opt_gpu(offspring, ind)
            
            # 全局负载均衡
            # for i in range(min(elite_size, len(self.population))):
            # for i in range(len(offspring)):
            n = len(offspring)
            for i in range(n):
                if random.random() < 0.2:
                    balanced_ind = offspring[i].copy()
                    self.load_balance(balanced_ind)
                    offspring.append(balanced_ind)
                elif random.random() <0.1:
                    self.distributed_individual_tasks(offspring, offspring[i])    
                
            
            # 限制子代大小
            # if len(offspring) > num_generations_all * 3:
            #     offspring = random.sample(offspring, num_generations_all * 3)
            
            # 评估子代适应度
            self.sch_data.Task_time_predictor.estimate_ga_population(
                offspring, self.sch_data.sch_task_list, all_node=True
            )
            
            offspring_scores = [self.fitness(ind) for ind in offspring]
            
            # 合并父代和子代，选择最佳个体
            combined = self.population + offspring
            combined_scores = scores + offspring_scores
            
            # 选择前N个最佳个体
            sorted_indices = np.argsort(combined_scores)[::-1]
            self.population = [combined[i] for i in sorted_indices[:num_generations_all]]
            scores = [combined_scores[i] for i in sorted_indices[:num_generations_all]]
            
            new_score = scores[0]
            self.write_log(f"Global optimization: New best score = {new_score}")
            self.write_log(f"best ind allocations: {self.population[0].task_array}")
            self.write_log(f"ind.completion_time: {self.population[0].completion_time}")
            self.write_log(f"ind.resource_area: {self.population[0].resource_area}")
            self.write_log(f"ind.total_runtime: {self.population[0].total_runtime}")
        
        # 选择最佳个体
        best_ind = max(self.population, key=lambda ind: ind.score)
        self.sch_data.best_ind = best_ind
        best_allocation = best_ind.task_array
        
        self.write_log("\nFinal Results:")
        self.write_log(f"Best individual score: {best_ind.score}")
        self.write_log(f"Best allocation: {best_allocation}")
        self.write_log(f"ind.completion_time: {best_ind.completion_time}")
        self.write_log(f"ind.resource_area: {best_ind.resource_area}")
        self.write_log(f"ind.total_runtime: {best_ind.total_runtime}")
        self.write_log(f"GA running time: {time.time() - start_time:.2f} seconds")
        
        logger.info("GA running time: %s seconds" % (time.time() - start_time))
        
        return best_allocation
    
    def run_ga_with_metrics(
            self,
            all_tasks,
            num_runs=50,
            num_generations_all=50,
            collector=None,
            operator_flags=None,
        ):
        """使用指标收集器的GA运行函数"""
        # 初始化数据收集器
        if collector is None:
            collector = GADataCollector()
        
        # 设置算子标志（控制哪些算子被使用）
        if operator_flags is None:
            operator_flags = {
                'crossover_pmx': True,
                'crossover_arith_ave': True,
                'mutate_seq': True,
                'mutate_resources': True,
                'opt1': True,
                'opt2': True,
                'opt_gpu': True,
                'load_balance': True,
                'distributed_tasks': True
            }
        
        start_time = time.time()
        self.write_log(f"\nStarting GA with {all_tasks} tasks on {self.resources} nodes")
        self.write_log(f"Active operators: {[op for op, flag in operator_flags.items() if flag]}")
        
        # 生成初始种群
        if operator_flags['distributed_tasks']:
            self.population = self.generate_balanced_population(all_tasks=all_tasks, population_size=num_generations_all)
        else:
            self.population = self.generate_population_all(all_tasks=all_tasks, population_size=num_generations_all)
        
        # 评估初始种群
        self.sch_data.Task_time_predictor.estimate_ga_population(
            self.population, self.sch_data.sch_task_list, all_node=True
        )
        
        # 计算初始种群的适应度
        scores = [self.fitness(ind) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]
        
        # 记录初始代
        collector.start_generation()
        collector.end_generation(0, self.population, scores)
        
        # GA主循环
        for global_gen in range(num_runs):
            collector.start_generation()
            self.write_log(f"\nGlobal Generation {global_gen + 1}")
            
            # 创建后代
            offspring = []
            
            # 精英保留
            elite_size = max(1, len(self.population) // 5)
            
            # 基本操作次数
            n_operations = len(self.population) * 2
            
            # 交叉操作
            for _ in range(n_operations):
                if len(self.population) > 1:
                    i, j = random.sample(range(len(self.population)), 2)
                    ind1, ind2 = self.population[i], self.population[j]
                    
                    if random.random() < 0.3:  # 交叉概率
                        # PMX交叉
                        if operator_flags['crossover_pmx'] and random.random() < 0.5:
                            old_len = len(offspring)
                            self.crossover_pmx(offspring, ind1, ind2)
                            
                            # 跟踪刚添加的个体
                            for new_ind in offspring[old_len:]:
                                collector.track_operator('crossover_pmx', new_ind)
                        
                        # 算术平均交叉
                        elif operator_flags['crossover_arith_ave']:
                            old_len = len(offspring)
                            self.crossover_arith_ave(offspring, ind1, ind2)
                            
                            # 跟踪刚添加的个体
                            for new_ind in offspring[old_len:]:
                                collector.track_operator('crossover_arith_ave', new_ind)
            
            # 变异操作
            for _ in range(n_operations):
                ind = random.choice(self.population)
                # print(ind.task_array)
                
                # 序列变异
                if operator_flags['mutate_seq'] and random.random() < 0.3 and len(ind.task_array)>=2:
                    old_len = len(offspring)
                    self.mutate_seq(offspring, ind)
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('mutate_seq', new_ind)
                
                # 资源变异
                if operator_flags['mutate_resources'] and random.random() < 0.3:
                    old_len = len(offspring)
                    self.mutate_resources(offspring, ind)
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('mutate_resources', new_ind)
            
            # 优化操作
            for _ in range(n_operations):
                ind = random.choice(self.population)
                
                # Opt1优化
                if operator_flags['opt1'] and random.random() < 0.7:
                    old_len = len(offspring)
                    self.opt1(offspring, ind)
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('opt1', new_ind)
                
                # Opt2优化
                if operator_flags['opt2'] and random.random() < 0.7:
                    old_len = len(offspring)
                    self.opt2(offspring, ind)
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('opt2', new_ind)
                
                # GPU优化
                if operator_flags['opt_gpu'] and random.random() < 0.7:
                    old_len = len(offspring)
                    self.opt_gpu(offspring, ind)
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('opt_gpu', new_ind)
            
            # 全局负载均衡
            for i in range(min(elite_size, len(self.population))):
                # 负载均衡
                if operator_flags['load_balance'] and random.random() < 0.2:
                    balanced_ind = self.population[i].copy()
                    collector.track_operator('load_balance')
                    self.load_balance(balanced_ind)
                    offspring.append(balanced_ind)
                    collector.track_operator('load_balance', balanced_ind)
                
                # 任务分配优化
                elif operator_flags['distributed_tasks'] and random.random() < 0.1:
                    old_len = len(offspring)
                    self.distributed_individual_tasks(offspring, self.population[i])
                    
                    # 跟踪刚添加的个体
                    for new_ind in offspring[old_len:]:
                        collector.track_operator('distributed_tasks', new_ind)
            
            # 评估后代适应度
            self.sch_data.Task_time_predictor.estimate_ga_population(
                offspring, self.sch_data.sch_task_list, all_node=True
            )
            
            offspring_scores = [self.fitness(ind) for ind in offspring]
            
            # 合并父代和子代，选择最佳个体
            combined = self.population + offspring
            combined_scores = scores + offspring_scores
            
            sorted_indices = np.argsort(combined_scores)[::-1]
            self.population = [combined[i] for i in sorted_indices[:num_generations_all]]
            scores = [combined_scores[i] for i in sorted_indices[:num_generations_all]]
            
            # 记录本代结束
            collector.end_generation(global_gen + 1, self.population, scores)
            
            self.write_log(f"Generation {global_gen + 1}: " +
                        f"Best score = {self.population[0].score:.4f}, " +
                        f"Avg = {np.mean(scores):.4f}")
        
        # 选择最佳个体
        best_ind = max(self.population, key=lambda ind: ind.score)
        self.sch_data.best_ind = best_ind
        best_allocation = best_ind.task_array
        
        # 完成运行时间
        metrics = collector.metrics
        metrics['total_runtime'] = time.time() - start_time
        metrics['active_operators'] = operator_flags
        
        self.write_log(f"\nFinal Results:")
        self.write_log(f"Best individual score: {best_ind.score:.4f}")
        self.write_log(f"GA running time: {metrics['total_runtime']:.2f} seconds")
        
        return best_allocation, collector