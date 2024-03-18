# import modules
from asyncio import futures
import copy
from ctypes import Union
from itertools import accumulate
from pathlib import Path
import logging
import re
import shutil
from collections import deque, OrderedDict
from typing import Collection, Dict, Any, Optional, ClassVar, Union, List
from copy import deepcopy
import json
from functools import partial, update_wrapper
import pandas as pd
import numpy as np
import time
import pickle
import random
import concurrent.futures
import sys
import heapq
from dataclasses import dataclass, field, asdict, is_dataclass
from colmena.models import Result


from  sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.append(r"/home/lizz_lab/cse30019698/project/colmena/multisite_/")
from my_util.data_structure import *

logger = logging.getLogger(__name__)

def dataclass_to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    else:
        return obj

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
    predict_run_seq: list = field(default_factory=list)

    # allocation information
    task_allocation: list[dict[str, int]] = field(default_factory=list)

    # initialize individual id
    def __post_init__(self):
        if self.individual_id == -1:
            self.individual_id = individual._next_id
            individual._next_id += 1

    # copy individual
    def copy(self):
        copied_individual = deepcopy(self)
        copied_individual.individual_id = individual._next_id
        individual._next_id += 1
        return copied_individual
    
    # hash
    def __hash__(self) -> int:
        sorted_allocation = sorted(self.task_allocation, key=lambda x: (x['name'], x['task_id']))
        return hash(str(sorted_allocation))
    
    def get_resources(self, task_name, task_id):
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

@dataclass
class available_task(SingletonClass):
    # task_names: list[str] = field(default_factory=list)
    # task_ids: list[dict[str, int]] = field(default_factory=list)
    # def __init__(self,task_names: set[str], task_ids: dict[str, int], task_datas=None):
    def __init__(self, task_ids: dict[str, list[str]]=None):
        # print(type(task_names))
        ## TODO task_names is set, but not work while get a str in it
        # self.task_names = set()
        # self.task_names = self.task_names.update(task_names)
        self.task_ids = task_ids
        
    def add_task_id(self, task_name:str, task_id:Union[str, list[str]]):
        # if task_name not in self.task_names:
        #     self.task_names.add(task_name)
        task_id = [task_id] if isinstance(task_id, str) else task_id
        for i in task_id:
            self.task_ids[task_name].append(i)
    
    def remove_task_id(self, task_name:str, task_id:Union[str, list[str]]):
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
    
    def get_task_nums(self):
        result = {}
        for key,value in self.task_ids.items():
            result[key] = len(value)
        return result
    
    def get_total_nums(self):
        return sum(self.get_task_nums().values())


@dataclass
class historical_data(SingletonClass):
    features = [
    "method",
    "message_sizes.inputs",
    # "worker_info.hostname",
    "resources.cpu",
    "resources.gpu",
    # "resources.thread",
    "time_running"]
    
    def __init__(self, methods: Collection[str], queue=None):
        self.historical_data = {}
        self.random_forest_model = {}
        from sklearnex import patch_sklearn, unpatch_sklearn
        patch_sklearn()
        for method in methods:
            self.historical_data[method] = []
            self.random_forest_model[method] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.queue = queue
    
    def add_data(self, feature_values: dict[str, Any]):
        method = feature_values['method']
        # if method not in self.historical_data:
        #     self.historical_data[method] = []
        #     self.random_forest_model[method] = RandomForestRegressor(n_estimators=100, random_state=42)
        if method not in self.historical_data:
            logger.warning(f"method {method} not in historical data")
        self.historical_data[method].append(feature_values)
    
    def get_features_from_result_object(self, result:Result):
        feature_values = {}
        for feature in self.features:
            value = result
            for key in feature.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                    break
                else:
                    value = getattr(value, key)
                # if value is None:
                #     break
            feature_values[feature] = value
        self.add_data(feature_values)
    
    def get_features_from_his_json(self, his_json:Union[str, list[str]]):
        for path in his_json:
            with open(path, 'r') as f:
                for line in f:
                    json_line = json.loads(line)
                    feature_values = {}
                    for feature in self.features:
                        value = json_line
                        for key in feature.split('.'):
                            value = value.get(key)
                            # if value is None:
                            #     break
                        feature_values[feature] = value
                    self.add_data(feature_values)
                    
    def random_forest_train(self):
        for method in self.historical_data:
            logger.info(f"train:{method} model")
            data = self.historical_data[method]
            df = pd.DataFrame(data)
            df.dropna(inplace=True)
            if len(data) < 5:
                continue
            model = self.random_forest_model[method]

            X = []
            y = []
            for feature_values in data:
                X = df.drop(columns=['time_running', 'method'])
                y = df['time_running']
            # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1.0, random_state=42)
            model.fit(X, y)
            logger.info(f"method: {method}, random forest regressor score: {model.score(X, y)}")
            # print(f"method: {method}, random forest regressor score: {model.score(X, y)}")

    def estimate_time(self, task):
        # use allocate resource to estimate running time
        method = task['name']
        result: Result = self.queue.result_list[task['task_id']]
        model = self.random_forest_model[method]
        feature_values = {}
        for feature in self.features:
            value = result
            for key in feature.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                    break
                else:
                    value = getattr(value, key)
            if key in task['resources']:
                value = task['resources'][key]
            feature_values[feature] = value
        X = pd.DataFrame([feature_values]).drop(columns=['time_running', 'method'])
        return model.predict(X)[0]
    
    # def estimate_batch(self, population):
    #     tasks = []
    #     for ind in population:
    #         task_allocation = ind.task_allocation
    #         tasks.extend(task_allocation)

    #     feature_values = []
    #     for task in tasks:
    #         result: Result = self.queue.result_list[task['task_id']]
    #         method = task['name']
    #         model = self.random_forest_model[method]
    #         feature_values.append((task, result, model))

    #     def estimate_task_time(task, result, model):
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
    #         X = pd.DataFrame([feature_values]).drop(columns=['time_running', 'method'])
    #         return model.predict(X)[0]

    #     # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     #     results = executor.map(lambda x: estimate_task_time(*x), feature_values)

    #     results = []
    #     for feature_value in feature_values:
    #         result = estimate_task_time(*feature_value)
    #         results.append(result)

    #     for task, runtime in zip(tasks, results):
    #         task['total_runtime'] = runtime
    #         # logger.info(f"predict runtime:{runtime}")

    def estimate_batch(self, population):
        def extract_feature_values(task, result):
            feature_values = {}
            for feature in self.features:
                value = result
                for key in feature.split('.'):
                    if isinstance(value, dict):
                        value = value.get(key)
                        break
                    else:
                        value = getattr(value, key)
                if key in task['resources']:
                    value = task['resources'][key]
                feature_values[feature] = value
            return feature_values
        
        tasks_by_model = {}  # 以模型为键，将任务分组存储
        for ind in population:
            task_allocation = ind.task_allocation
            for task in task_allocation:
                model_name = task['name']
                if model_name not in tasks_by_model:
                    tasks_by_model[model_name] = []
                tasks_by_model[model_name].append(task)

        for model_name, tasks in tasks_by_model.items():
            model = self.random_forest_model[model_name]  # 获取对应的模型

            feature_values = []
            for task in tasks:
                result: Result = self.queue.result_list[task['task_id']]
                feature_dict = extract_feature_values(task,result)
                feature_values.append(feature_dict)

            X = pd.DataFrame(feature_values).drop(columns=['time_running', 'method'])
            predictions = model.predict(X)

            for task, runtime in zip(tasks, predictions):
                task['total_runtime'] = runtime
                # logger.info(f"Predicted runtime for task {task['task_id']}: {runtime}")

    # def estimate_batch(self, population):
    #     # compare to estimate time cost
    #     tasks = []
    #     for ind in population:
    #         task_allocation = ind.task_allocation
    #         tasks.extend(task_allocation)
    #     for task in tasks:
    #         task['total_runtime'] = 10
            


class evosch2:
    """add all task in individual
    cost function calculate total time idle time
    """
    def __init__(self,  resources=None,at:available_task=None, hist_data=historical_data, population_size=10):
        self.his_population = set()
        # self.task_queue_audit = task_queue_audit ## this is hist data, not need it in evosch2, introduce in colmena queue while trigger schduling
        self.resources:dict = resources # this is for management
        self.resources_evo:dict = resources # this is for evo sch
        self.hist_data = hist_data
        self.at = at # available task
        # self.population = self.generate_population(population_size)
        self.population = []
        
        ## log the running task for track the resource and time
        self.running_task:list[dict[str, int]] = [] # {'task_id': 1, 'name': 'simulate', 'start_time': 100, 'finish_time': 200, 'total_time': 100, resources:{'cpu':3,'gpu':0}}
        self.current_time = 0 # current running time for compute  while trigger evo_scheduler
        ## add for restore resources
        # self.current_time # current running time for compute when the resources back to pool again
        # self.restore_resources = {} # {'resources':{'cpu':3,'gpu':0}, 'restore_time': 100}
    
    ## TODO estimate runtime background and generate new individual
    
    def get_task_nums(self):
        pass
    def get_resources(self):
        return self.resources
    def get_piority(self):
        pass

    def detect_submit_sequence(self):

        # detect submit task sequence to choose proper task to run
        pass
    
    def detect_no_his_task(self):
        '''
        detect if there is task not in historical data
        without historical data, estimate method cannot work, we need run them first, and record the historical data to train the model
        return the individual contain no record task
        # get all node resource or get user predifine resource
        '''
        task_queue = []
        predict_running_seq = []
        all_tasks = self.at.get_all()
        for name, ids in all_tasks.items():
            if len(ids) > 0:
                if len(self.hist_data.historical_data[name])<5: # no data for train
                    cpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'cpu')
                    new_task = {
                        "name":name,
                        "task_id": ids[0],
                        "resources":{
                            # "cpu": self.get_resources()['cpu']
                            "cpu": cpu
                        }
                    }
                    task_queue.append(new_task)
                    predict_running_seq.append({
                        'name': name,
                        'task_id': ids[0],
                        'start_time': None,
                        'finish_time': None,  
                        'total_runtime': 10 ** 7, # 无意义值
                        'resources':{
                            'cpu': cpu
                        }  
                    })
        ind = individual(tasks_nums=len(task_queue),total_resources=self.get_resources())
        ind.task_allocation = task_queue
        ind.predict_run_seq = predict_running_seq
        return ind


    # def calculate_completion_time(self, ind:individual):
    #     available_cpu = self.resources_evo['cpu']
    #     current_time = 0
    #     ongoing_task = []
    #     for task in ind.task_allocation:
    #         while ongoing_task and ongoing_task[0][0] <= current_time:
    #             _, cpus = heapq.heappop(ongoing_task)
    #             available_cpu += cpus
                
    #         while available_cpu < task['resources']['cpu']:
    #             if ongoing_task:
    #                 next_finish_time, cpus = heapq.heappop(ongoing_task)
    #                 current_time = next_finish_time
    #                 available_cpu += cpus
    #             else:
    #                 break
            
    #         if available_cpu < task['resources']['cpu']:
    #             raise ValueError("Not enough CPUs for all tasks")
            
    #         available_cpu -= task['resources']['cpu']
    #         # finish_time = current_time + self.hist_data.estimate_time(task = task)
    #         finish_time = current_time + task['total_runtime']
    #         heapq.heappush(ongoing_task, (finish_time, task['resources']['cpu']))
    #     while ongoing_task:
    #         next_finish_time, cpus = heapq.heappop(ongoing_task)
    #         available_cpu += cpus
    #         current_time = next_finish_time
    #     return current_time
    
    # def calculate_completion_time_record(self, ind):
    #     available_cpu = self.resources_evo['cpu']
    #     current_time = 0
    #     ongoing_task = []
    #     running_seq = []  # 记录任务执行的顺序和时间

    #     for task in ind.task_allocation:
    #         # 检查是否有任务已经完成
    #         while ongoing_task and ongoing_task[0][0] <= current_time:
    #             _, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
    #             available_cpu += cpus
    #             # 更新任务的完成时间
    #             for task_record in running_seq:
    #                 if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
    #                     task_record['finish_time'] = current_time
    #                     task_record['total_runtime'] = current_time - task_record['start_time']
    #                     break

    #         # 等待直到有足够的CPU资源
    #         while available_cpu < task['resources']['cpu']:
    #             if ongoing_task:
    #                 next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
    #                 for task_record in running_seq:
    #                     if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
    #                         task_record['finish_time'] = current_time
    #                         task_record['total_runtime'] = current_time - task_record['start_time']
    #                         break
    #                 current_time = next_finish_time
    #                 available_cpu += cpus
    #             else:
    #                 break

    #         if available_cpu < task['resources']['cpu']:
    #             # skip current task
    #             raise ValueError("Not enough CPUs for all tasks")
    #             # logger.info("Not enough CPUs for all tasks")
            
    #         # start a new one
    #         available_cpu -= task['resources']['cpu']
    #         start_time = current_time
    #         # finish_time = current_time + self.hist_data.estimate_time(task = task)
    #         finish_time = current_time + task['total_runtime']
    #         heapq.heappush(ongoing_task, (finish_time, task['resources']['cpu'], task['task_id'], task['name']))

    #         # 记录任务的开始时间和其他信息
    #         running_seq.append({
    #             'name': task['name'],
    #             'task_id': task['task_id'],
    #             'start_time': start_time,
    #             'finish_time': None,  # 将在任务完成时更新
    #             'total_runtime': None  # 将在任务完成时更新
    #         })

    #     # 清空剩余的任务并记录完成时间
    #     while ongoing_task:
    #         next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
    #         available_cpu += cpus
    #         current_time = next_finish_time
    #         # 更新任务的完成时间
    #         for task_record in running_seq:
    #             if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
    #                 task_record['finish_time'] = current_time
    #                 task_record['total_runtime'] = current_time - task_record['start_time']
    #                 break
        
    #     ind.predict_run_seq = running_seq
    #     # 返回总完成时间和任务运行序列
    #     return current_time
    
    def calculate_completion_time_record_with_running_task(self, ind):
        # TODO need change for multinode and heterogenous
        available_cpu = self.resources_evo['cpu']
        current_time = time.time()
        record = current_time
        ongoing_task = []
        running_seq = []  # 记录任务执行的顺序和时间
        
        ## consider already submit task
        if self.running_task:
            for task in self.running_task:
                heapq.heappush(ongoing_task, (task['finish_time'], task['resources']['cpu'], task['task_id'], task['name']))
                available_cpu -= task['resources']['cpu']
                running_seq.append({
                    'name': task['name'],
                    'task_id': task['task_id'],
                    'start_time': task['start_time'],
                    'finish_time': task['finish_time'],  # 将在任务完成时更新
                    'total_runtime': task['total_runtime'],  # 将在任务完成时更新
                    'resources': task['resources']
                })

        for task in ind.task_allocation:
            # 检查是否有任务已经完成
            while ongoing_task and ongoing_task[0][0] <= current_time:
                _, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
                available_cpu += cpus
                # 更新任务的完成时间
                for task_record in running_seq:
                    if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                        task_record['finish_time'] = current_time
                        task_record['total_runtime'] = current_time - task_record['start_time']
                        break

            # 等待直到有足够的CPU资源
            while available_cpu < task['resources']['cpu']:
                if ongoing_task:
                    next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
                    for task_record in running_seq:
                        if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                            task_record['finish_time'] = current_time
                            task_record['total_runtime'] = current_time - task_record['start_time']
                            break
                    current_time = next_finish_time
                    available_cpu += cpus
                else:
                    break

            if available_cpu < task['resources']['cpu']:
                # skip current task
                raise ValueError("Not enough CPUs for all tasks")
                # logger.info("Not enough CPUs for all tasks")
            
            # start a new one
            available_cpu -= task['resources']['cpu']
            start_time = current_time
            # finish_time = current_time + self.hist_data.estimate_time(task = task)
            finish_time = current_time + task['total_runtime']
            heapq.heappush(ongoing_task, (finish_time, task['resources']['cpu'], task['task_id'], task['name']))

            # 记录任务的开始时间和其他信息
            running_seq.append({
                'name': task['name'],
                'task_id': task['task_id'],
                'start_time': start_time,
                'finish_time': None,  # 将在任务完成时更新
                'total_runtime': None,  # 将在任务完成时更新
                'resources': task['resources']
            })

        # 清空剩余的任务并记录完成时间
        while ongoing_task:
            next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
            available_cpu += cpus
            current_time = next_finish_time
            # 更新任务的完成时间
            for task_record in running_seq:
                if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                    task_record['finish_time'] = current_time
                    task_record['total_runtime'] = current_time - task_record['start_time']
                    break
        
        ind.predict_run_seq = running_seq
        # 返回总完成时间和任务运行序列
        return current_time-record
    
    def calculate_total_time(self, ind:individual):
        total_time = 0
        for task in ind.task_allocation:
            # total_time += self.hist_data.estimate_time(task)
            total_time += task['total_runtime']
        return total_time
            
    
    
    def fitness(self, ind:individual):
        # calculate total time based on avail resources and task
        total_time = 0  ## time accumulate by all task
        completion_time = 0 ## HPC makespan

        total_time = self.calculate_total_time(ind)
        completion_time = self.calculate_completion_time_record_with_running_task(ind)
        
        # ind.score = 1000/completion_time
        ind.score = -completion_time
        return ind.score
        
    
    def initialize_individual(self):
        pass
    
    def generate_population(self, population_size: int):
        ## add all task to individual
        task_nums = self.at.get_task_nums()
        all_tasks = self.at.get_all()
        population = []
        if self.resources_evo['cpu']>16:
            for _ in range(population_size//3+1):
                ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
                
                task_queue = []
                for name, ids in all_tasks.items():
                    for task_id in ids:
                        new_task = {
                            "name":name,
                            "task_id": task_id,
                            "resources":{
                                "cpu": random.randint(1,16)
                                # "cpu": 1
                            }
                        }
                        task_queue.append(new_task)
                random.shuffle(task_queue)
            
                ind.task_allocation = task_queue
                population.append(ind)
            
        for _ in range(population_size//3+1):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
            
            task_queue = []
            for name, ids in all_tasks.items():
                for task_id in ids:
                    new_task = {
                        "name":name,
                        "task_id": task_id,
                        "resources":{
                            # "cpu": random.randint(1,16)
                            "cpu": 1
                        }
                    }
                    task_queue.append(new_task)
            random.shuffle(task_queue)
        
            ind.task_allocation = task_queue
            population.append(ind)
        
        for _ in range(population_size//3+1):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
            
            task_queue = []
            for name, ids in all_tasks.items():
                for task_id in ids:
                    cpu = getattr(self.hist_data.queue.result_list[task_id].resources,'cpu')
                    new_task = {
                        "name":name,
                        "task_id": task_id,
                        "resources":{
                            # "cpu": random.randint(1,16)
                            "cpu": cpu
                        }
                    }
                    task_queue.append(new_task)
            random.shuffle(task_queue)
        
            ind.task_allocation = task_queue
            population.append(ind)
            
        return population 
    
    def mutate_cpu(self,ind:individual):
        ## change resource 
        alloc = random.choice(ind.task_allocation)
        choice = [-5,-3,-2,-1,1,2,3,5]
        new_alloc = alloc['resources']['cpu'] + random.choice(choice)
        
        if new_alloc <= 0:
            alloc['resources']['cpu'] = 1
        else:
            alloc['resources']['cpu'] = new_alloc
    
    def mutate_seq(self,ind:individual):
        ## change task sequence
        index1 = random.randrange(len(ind.task_allocation))
        index2 = random.randrange(len(ind.task_allocation))
        while index2 == index1:
            index2 = random.randrange(len(ind.task_allocation))
            
        ind.task_allocation[index1], ind.task_allocation[index2] = ind.task_allocation[index2], ind.task_allocation[index1]
    
    
    def crossover_arith_ave(self, ind1:individual, ind2:individual):
        task_avg = [None]*len(ind1.task_allocation)
        for i in range(len(ind1.task_allocation)):
            name = ind1.task_allocation[i]['name']
            task_id = ind1.task_allocation[i]['task_id']
            task_avg[i] = {
                "name": name,
                "task_id": task_id,
                "resources":{
                    "cpu": (ind1.get_resources(name,task_id)['cpu']+ind2.get_resources(name,task_id)['cpu'])//2
            }}
        ind1.task_allocation = task_avg
    
    def list_dict_found(self, list_dic, dic):
        for i in range(len(list_dic)):
            if list_dic[i]['task_id'] == dic['task_id'] and list_dic[i]['name']== dic['name']:
                return True
        return False

    def list_dict_index(self, list_dic, dic):
        for i in range(len(list_dic)):
            if list_dic[i]['task_id'] == dic['task_id'] and list_dic[i]['name']== dic['name']:
                return i
        return None

    def crossover_pmx(self, ind:individual, ind2:individual):
        size = len(ind.task_allocation)
        p1, p2 = [0]*size, [0]*size
        
        cxpoint1 = random.randint(0, size-1)
        cxpoint2 = random.randint(0, size-1)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
        # print(cxpoint1,cxpoint2)
        for i in range(cxpoint1,cxpoint2+1):
            p1[i] = ind2.task_allocation[i]
            p2[i] = ind.task_allocation[i]
            
        for i in range(size):
            if i < cxpoint1 or i > cxpoint2:
                ii = ind.task_allocation[i]
                while self.list_dict_found(p1[cxpoint1:cxpoint2+1],ii):
                    # ii = ind.task_allocation[p1[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind.task_allocation[self.list_dict_index(ind2.task_allocation,ii)]
                p1[i] = ii
                
                ii = ind2.task_allocation[i]
                while self.list_dict_found(p2[cxpoint1:cxpoint2+1],ii):
                    # ii = ind2.task_allocation[p2[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind2.task_allocation[self.list_dict_index(ind.task_allocation,ii)]
                p2[i] = ii
        
        ind.task_allocation = p1
        ind2.task_allocation = p2
        
    def opt1(self, ind:individual):
        ## add resources for longest task
        # logger.info(f"opt1: {ind.task_allocation}")
        # logger.info(f"opt1: {ind.predict_run_seq}")
        task = max(ind.predict_run_seq, key=lambda x:x['total_runtime'])
        
        index = self.list_dict_index(ind.task_allocation,task)
        # task may in running, so we need to check if it is in the task allocation
        if index:
            new_alloc = random.choice([1,2,3,4,5]) + ind.task_allocation[index]['resources']['cpu']
            if new_alloc <= self.resources_evo['cpu']//2: # only allow at constrait resources
                ind.task_allocation[index]['resources']['cpu'] = new_alloc
        
    def opt2(self, ind:individual):
        ## advance the latest task order
        task = max(ind.predict_run_seq, key=lambda x:x['finish_time'])
        index = self.list_dict_index(ind.task_allocation,task)
        # task may in running, so we need to check if it is in the task allocation
        if index:
            new_index = random.randrange(0, index)
            element = ind.task_allocation.pop(index)
            ind.task_allocation.insert(new_index, element)
            
    def process_individual(self,ind1,ind2,crossover_rate, mutation_rate):
        # logger.info(f"process_infividual:{ind1.individual_id}")
        if random.random() < 0.5:
            if random.random() < mutation_rate:
                self.mutate_cpu(ind1)
            elif random.random() < mutation_rate:
                self.mutate_seq(ind1)
                
            if random.random() < crossover_rate/2:
                self.crossover_pmx(ind1,ind2)
            elif random.random() < crossover_rate/2:
                self.crossover_arith_ave(ind1,ind2)
            
        else:
            self.opt1(ind1)
            self.opt2(ind1)
        
            
    def run_ga(self, num_generations):
        # run no record task
        ind = self.detect_no_his_task()
        if len(ind.task_allocation) > 0:
            return ind
        
        # train random forest model for predict running time
        self.hist_data.random_forest_train()

        # resources = self.get_resources()
        # population = self.generate_population(pop_size)
        logger.info(f"Starting GA with available tasks: {self.at.get_all()}")
        pop_size = len(self.population)
        population = self.population

        # self.his_population.update(population)
        # first we can estimate every task running time
        self.hist_data.estimate_batch(population)
        scores = [self.fitness(ind) for ind in population]
        population = [population[i] for i in np.argsort(scores)[::-1]]
        # only one task , submit directly
        if self.at.get_total_nums() == 1:
            logger.info(f"Only one task, submit directly")
            return population[0]
        # logger.info(f"Generation 0: {population[0]}")
        for gen in range(num_generations):
            # population=population[::-1]
            # population = population[pop_size // 2:] + [ind.copy() for ind in population[pop_size // 2:]]
            population.extend([ind.copy() for ind in population[:pop_size // 2]])
            next_population = population[:pop_size // 2]
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                size = len(next_population)
                for i in range(size//2):
                    futures.append(executor.submit(self.process_individual(next_population[i],next_population[size-i-1],0.8,0.8)))
            concurrent.futures.wait(futures)
            # size = len(next_population)
            # for i in range(size // 2):
            #     self.process_individual(next_population[i], next_population[size - i - 1], 0.8, 0.8)
            
            self.hist_data.estimate_batch(population)
            scores = [self.fitness(ind) for ind in population]
            population = [population[i] for i in np.argsort(scores)[::-1]]
            logger.info(f"Generation {gen}: {population[0].score}")
            population = population[:pop_size]
        best_ind = max(population, key=lambda ind: ind.score)
        logger.info(f"score of all ind:{scores}")
        logger.info(f"Best ind:{best_ind}")
        return best_ind
        
    


    
## test evo_sch
if __name__ == '__main__':
    ## temp test
    out_dir = Path('/home/yxx/work/project/colmena/multisite_/my_test/ga_simulation_test')
    with open(out_dir / 'task_queue_audit.pkl', 'rb') as f:
        task_queue_audit = pickle.load(f)
    with open(out_dir / 'length_time', 'rb') as fp:
        length_times = pickle.load(fp)
    with open(out_dir / 'cpu_time', 'rb') as fp:
        core_times = pickle.load(fp)
    total_cpu = 64

    trainning_time = 100
    sampling_time = 100
    inference_time = 100
    # topics=['train', 'sample', 'infer', 'simulate']
    methods = {'train', 'evaluate', 'run_calculator', 'run_sampling'}
    hist_data = historical_data(methods=methods)
    tasks = {}
    for method in methods:
        tasks[method] = []
    available_task__ = available_task(task_ids=tasks)
    available_resources = {"cpu": 64, "gpu": 4, "memory": "128G"}
    sch = evosch2(resources=available_resources, at=available_task__, hist_data=hist_data)
    sch.hist_data.add_data('length_times', length_times)
    sch.hist_data.add_data('core_times', core_times)
    print(sch.hist_data.historical_data['length_times'])
    print(sch.hist_data.historical_data['core_times'])
    
    from fff.simulation.utils import write_to_string
    with open('/home/yxx/work/project/colmena/multisite_/my_test/colmena_test/task_queue_audit', 'rb') as f:
        task_queue_audit = pickle.load(f)
    for idx, task in enumerate(task_queue_audit[0:20]):
        sch.at.add_task_id('simulate', str(idx))
    
    population = sch.generate_population(10)
    # print(population[0].task_allocation)
    sch.population = population
    ind = sch.run_ga(10)
    print(ind.task_allocation)
    # ind.save_to_json('/home/yxx/work/project/colmena/multisite_/my_test/colmena_test/colmena_test/sch_result.json')
    print(len(ind.task_allocation))
    print(sch.resources)
    for task in ind.task_allocation[0:3]:
        task = ind.task_allocation.pop(0)
        sch.at.remove_task_id(task['name'], task['task_id'])
        
        for key,value in task['resources'].items():
            sch.resources[key]-=value
            
        print(sch.resources)
        
    for idx, task in enumerate(task_queue_audit[20:25],start=20):
        sch.at.add_task_id('simulate', str(idx))
    ind = sch.run_ga(10)
    print(ind.task_allocation)

        
    
        
        