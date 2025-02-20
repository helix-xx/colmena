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

# 请修改下面函数，除了running和task两个队列外，还需要增加queued_task。其中queued task为先前分配了执行顺序和资源的任务，但还未执行。请你基于下面函数，增加queud_task任务到模拟的任务运行队列中。queued task和task队列一样，有cpu资源使用量和gpu资源使用两和queued_task_runtime信息

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
    
    # 处理新任务
    for i in range(n_tasks):
        required_cpu = task_cpu[i]
        required_gpu = task_gpu[i]
        duration = task_runtime[i]
        
        # 检查已完成任务
        while task_count > 0 and new_ongoing_times[0] <= current_time:
            avail_cpu += new_ongoing_cpus[0]
            avail_gpu += new_ongoing_gpus[0]
            
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
        
    # 计算空闲资源面积
    resources_released_weighted = 0
    resources_released_weighted += (avail_cpu + avail_gpu) * current_time
    while task_count > 0:
        current_time = new_ongoing_times[0]
        avail_cpu += new_ongoing_cpus[0]
        avail_gpu += new_ongoing_gpus[0]
        resources_released_weighted += (avail_cpu + avail_gpu) * current_time
        
        # 移除完成的任务
        for j in range(task_count - 1):
            new_ongoing_times[j] = new_ongoing_times[j + 1]
            new_ongoing_cpus[j] = new_ongoing_cpus[j + 1]
            new_ongoing_gpus[j] = new_ongoing_gpus[j + 1]
        task_count -= 1
    
    resources_released_weighted = current_time * (avail_cpu + avail_gpu) - resources_released_weighted
    
    completion_time = current_time - start_time if n_tasks > 0 else 0
    return completion_time, resources_released_weighted, task_starts, task_ends


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
        total_runtime = defaultdict(float)
        
        for node in self.node_resources.keys():
            node_mask = ind.task_array['node'] == node
            node_tasks = ind.task_array[node_mask]
            
            total_cpu_time[node], total_gpu_time[node] = self.calc_time(node_tasks)
            
            completion_time[node], _, total_runtime[node] = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
                node_tasks,
                ind
            )

        return total_cpu_time, total_gpu_time, completion_time, total_runtime

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


    def calculate_completion_time_record_with_running_task(self, resources, task_array, ind):
        """计算完成时间并更新任务时间记录"""
        if len(task_array) == 0:
            return 0.0, 0, 0.0
        node = task_array[0]['node']
        if self.fixed_state[node] is None:
            raise ValueError("Fixed state not calculated")
        
        # 提取简单数组
        task_cpu = np.array([task['cpu'] for task in task_array], dtype=np.int32)
        task_gpu = np.array([task['gpu'] for task in task_array], dtype=np.int32)
        task_runtime = np.array([task['total_runtime'] for task in task_array], dtype=np.float64)
        
        # 使用预计算状态计算完成时间
        try:
            completion_time, resources_released_weighted, starts, ends = _calculate_completion_time_with_state(
                task_cpu,
                task_gpu,
                task_runtime,
                self.fixed_state[node][0],  # current_time
                self.fixed_state[node][1],  # avail_cpu
                self.fixed_state[node][2],  # avail_gpu
                self.fixed_state[node][3],  # task_count
                self.fixed_state[node][4],  # ongoing_times
                self.fixed_state[node][5],  # ongoing_cpus
                self.fixed_state[node][6],  # ongoing_gpus
                resources['cpu'],
                resources['gpu']
            )
        except Exception as e:
            print(f"Error occurred during resource allocation: {e}")
            print(self.fixed_state[node], task_cpu,task_gpu,task_runtime)
            raise e
        
        if completion_time <= 0:
            print(task_array)
            print(self.fixed_state)
            raise ValueError("Resource allocation failed")
        
        # 计算资源使用面积    
        resource_area = _calculate_task_resource_area(task_runtime, task_cpu, task_gpu)
            
        # 更新individual中的任务时间
        for i, task in enumerate(task_array):
            idx = ind._task_id_index[task['task_id']]
            ind.task_array[idx]['start_time'] = starts[i]
            ind.task_array[idx]['finish_time'] = ends[i]
            
        total_runtime = np.max(ends)
            
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
            # # 添加空任务检查
            # if len(node_tasks) == 0:
            #     completion_times[i] = 0
            #     resource_areas[i] = 0
            #     total_runtimes[i] = 0
            #     continue
            
            completion_time, resource_area, total_runtime = self.calculate_completion_time_record_with_running_task(
                self.node_resources[node],
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
                if task_nums == 1:
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
        
    def precalculate_fixed_state(self, running_tasks_all, queued_tasks_all):
        """预计算固定任务状态"""
        self.fixed_state = {}
        for node in self.node_resources.keys():
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
            self.fixed_state[node] = _precalculate_fixed_tasks_state(
                queued_task_cpu,
                queued_task_gpu,
                queued_task_runtime,
                running_finish_times,
                running_cpus,
                running_gpus,
                self.node_resources[node]['cpu'],
                self.node_resources[node]['gpu'],
                0
            )
            
        if self.fixed_state is None:
            raise ValueError("Fixed tasks resource allocation failed")
        
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
        
        self.precalculate_fixed_state(self.running_task_node, self.sch_data.avail_task.allocations)
        
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

        # only one task , submit directly
        # if self.at.get_total_nums(all_tasks) == 1:
        #     logger.info(f"Only one task, submit directly")
        #     self.best_ind = self.population[-1]  # predifined resources at last in list
        #     return self.best_ind.task_array

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
        # best_allocation = best_ind.task_allocation
        best_allocation = best_ind.task_array
        self.write_log("\nFinal Results:")
        self.write_log(f"scores of all individuals: {scores}")
        self.write_log(f"Best individual score: {best_ind.score}")
        self.write_log(f"Best allocation: {best_allocation}")
        self.write_log(f"GA running time: {time.time() - start_time:.2f} seconds")


        logger.info("GA running time: %s seconds" % (time.time() - start_time))
        # self.at.move_allocation_to_scheduled(all_tasks, best_allocation) # should consider lock
        return best_allocation