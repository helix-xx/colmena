# import modules
from asyncio import futures
import copy
from ctypes import Union
from itertools import accumulate
from pathlib import Path
import logging
import re
import shutil
from collections import deque, OrderedDict, defaultdict
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
import os
import psutil
import gc
import heapq
from dataclasses import dataclass, field, asdict, is_dataclass
from colmena.models import Result


from  sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

relative_path = "~/project/colmena/multisite_"
absolute_path = os.path.expanduser(relative_path)
sys.path.append(absolute_path)
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
    predict_run_seq: list = field(default_factory=list) # [task,,,]
    predict_run_seq_node: dict = field(default_factory=dict)#{'node':{'task_id':task,,,},,,}

    # allocation information
    task_allocation: list[dict[str, int]] = field(default_factory=list) # store all task
    task_allocation_node: dict = field(default_factory=dict) # store task on each node

    # initialize individual id
    def __post_init__(self):
        if self.individual_id == -1:
            self.individual_id = individual._next_id
            individual._next_id += 1

    # deepcopy individual
    def copy(self):
        copied_individual = deepcopy(self)
        copied_individual.individual_id = individual._next_id
        individual._next_id += 1
        return copied_individual
    
    # hash
    def __hash__(self) -> int:
        sorted_allocation = sorted(self.task_allocation, key=lambda x: (x['name'], x['task_id']))
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
        for key,value in self.task_ids.items(): #key: task name, value:task id list
            result[key] = len(value)
        return result
    
    def get_total_nums(self):
        return sum(self.get_task_nums().values())


@dataclass
class historical_data(SingletonClass):
    features = [
    "method",
    "message_sizes.inputs",
    # "worker_info.hostname", # for multinode executor # need vectorize for training
    "resources.cpu",
    "resources.gpu",
    # "resources.thread",
    "time_running"]
    
    def __init__(self, methods: Collection[str], queue=None):
        # submit history and complete history for cal the potential improvement
        self.methods = methods
        self.submit_task_seq = []
        self.complete_task_seq = []
        ## total submit / total complete
        self.trigger_info = []
        
        # his data for predict time runnint 
        self.historical_data = {}
        self.random_forest_model = {}
        from sklearnex import patch_sklearn, unpatch_sklearn
        patch_sklearn()
        for method in self.methods:
            self.historical_data[method] = []
            self.random_forest_model[method] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.queue = queue
    
    def get_child_task(self, methods):
        submit_task_seq = deepcopy(self.submit_task_seq)
        complete_task_seq = deepcopy(self.complete_task_seq)
        whole_seq = submit_task_seq + complete_task_seq
        whole_seq.sort(key=lambda x:x['time'])
        now_submit = {}
        pre_complete = {}
        has_submit = False
        has_complete = False
        def initialize():

            for method in methods:
                now_submit[method]=0
                pre_complete[method]=0
        def record():
            self.trigger_info.append({
                "submit": deepcopy(now_submit),
                "complete": deepcopy(pre_complete)
            })
            initialize()
        
        initialize()
            
        while(whole_seq):
            task = whole_seq.pop()
            if task["type"]=="submit":
                if has_submit and has_complete:
                    record()
                    has_complete = False
                now_submit[task["method"]]+=1
                has_submit = True
            elif has_submit:
                pre_complete[task["method"]]+=1
                has_complete = True
            else:
                # no submit task after complete task
                # just pass
                pass
         # After processing all tasks, record any remaining tasks
        if has_submit and has_complete:
            record()
        
        return self.trigger_info
    
    # get the last trigger info until all method has trigger submit task
    def get_closest_trigger_info(self):
        cloest_trigger_info = []
        method_flag = {}
        for method in self.methods:
            method_flag[method] = False
        for info in self.trigger_info:
            cloest_trigger_info.append(info)
            for method in self.methods:
                if info["submit"][method] > 0:
                    method_flag[method] = True
            if all(method_flag.values()):
                return cloest_trigger_info
    
    def get_child_task_time(self, method):
        trigger_info = self.get_closest_trigger_info()
        info = trigger_info.pop()
        # get each task info from historyical data
        # add to the individual task allocation 
        # TODO
        pass
    
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

    def estimate_batch(self, population, all_node=False):
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
            


class evosch2:
    """add all task in individual
    cost function calculate total time idle time
    """
    def __init__(self,  resources:dict = None,at:available_task=None, hist_data=historical_data, population_size=10):
        # self.his_population = set() # should following round consider history evo result?
        self.node_resources:dict = resources
        self.resources:dict = self.node_resources # used in colmena base.py.  data: {"node1":{"cpu":56,"gpu":4},"node2":{"cpu":56,"gpu":4}}
        for key,value in self.resources.items():
            value['gpu_devices']=list(range(value['gpu'])) #TODO gpu nums change to gpu_devices ids; next we need get ids from config
        self.resources_evo:dict = self.resources # used in evosch
        logger.info("total resources: {self.resources_evo}")
        self.hist_data = hist_data
        self.at = at # available task
        self.population = [] # [individual,,,] # store all individual on single node
        self.population_node = defaultdict(list) # {node: [individual,,,],,,} # store all individual on all node
        
        ## log the running task for track the resource and time
        self.running_task:list[dict[str, int]] = [] # {'task_id': 1, 'name': 'simulate', 'start_time': 100, 'finish_time': 200, 'total_time': 100, resources:{'cpu':3,'gpu':0}}
        self.running_task_node:dict = defaultdict(list)
        self.current_time = 0 # current running time for compute  while trigger evo_scheduler
        self.best_ind:individual = None
    
    ## TODO estimate runtime background and generate new individual
    
    def get_dict_list_nums(self,dict_list:dict):
        """
        get total task nums from task allocation
        """
        task_nums = 0
        for key,value in dict_list.items():
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
            if len(pending_queue[node]) == 0: # no pending task on node
                return True
        
        return False # all node have task pending
    
    def get_piority(self):
        pass

    def cpu_choice(self):
        pass
    
    def gpu_choice(self):
        pass

    def detect_submit_sequence(self):
        # detect submit task sequence to choose proper task to run
        pass
    
    def detect_no_his_task(self, total_nums = 5):
        '''
        detect if there is task not in historical data
        without historical data, estimate method cannot work, we need run them first, and record the historical data to train the model
        return the individual contain no record task
        # get all node resource or get user predifine resource
        '''
        task_queue = defaultdict(list)
        predict_running_seq = defaultdict(list)
        all_tasks = self.at.get_all()
        which_node = self.generate_node()
        for name, ids in all_tasks.items():
            avail = len(ids)
            hist = len(self.hist_data.historical_data[name]) # we dont have historical data for estimate
            if (hist < total_nums) and (avail > 0):
                # cpu choices
                predefine_cpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'cpu')
                cpu_lower_bound = min(2, predefine_cpu//2)
                cpu_upper_bound = max(self.node_resources.values(), key=lambda x: x['cpu'])['cpu'] # node max cpu value
                sample_nums = min((total_nums - hist),avail)
                choices = np.linspace(cpu_lower_bound, cpu_upper_bound, num=sample_nums, endpoint=True, retstep=False, dtype=int)
                
                # gpu choices
                predefine_gpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'gpu')
                if predefine_gpu == 0: # this task do not use gpu
                    gpu_choices = np.linspace(0, 0, num=sample_nums, endpoint=True, retstep=False, dtype=int)
                else:
                    gpu_lower_bound = 1
                    gpu_upper_bound = max(self.node_resources.values(), key=lambda x: x['gpu'])['gpu']
                    gpu_choices = np.linspace(gpu_lower_bound, gpu_upper_bound, num=sample_nums, endpoint=True, retstep=False, dtype=int)
                # if len(self.hist_data.historical_data[name])<5: # no data for train
                for i in range(len(choices)):
                    cpu = int(choices[i])
                    gpu = int(gpu_choices[i])
                    node = next(which_node)
                    new_task = {
                        "name":name,
                        "task_id": ids[i],
                        "resources":{
                            "cpu": cpu,
                            "gpu": gpu,
                            "node": node
                        },
                        "total_runtime": 1
                    }
                    task_queue[node].append(new_task)
                    predict_running_seq[node].append({
                        'name': name,
                        'task_id': ids[i],
                        'start_time': None,
                        'finish_time': None,  
                        'total_runtime': 1, # 无意义值
                        'resources':{
                            'cpu': cpu,
                            'gpu': gpu,
                            "node": node
                        }  
                    })
        ind = individual(tasks_nums=self.get_dict_list_nums(task_queue),total_resources=copy.deepcopy(self.get_resources()))
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
    
    def calc_utilization(self,ind):
        total_cpu_time = defaultdict(int)
        total_gpu_time = defaultdict(int)
        completion_time = defaultdict(int)
        for node in self.node_resources.keys():
            total_cpu_time[node], total_gpu_time[node] = self.calc_time(ind.task_allocation_node[node], node)
            completion_time[node], running_swq =  self.calculate_completion_time_record_with_running_task(self.node_resources[node], self.running_task_node[node], ind.task_allocation_node[node])
        
        return total_cpu_time, total_gpu_time, completion_time
    
    def load_balance(self, ind):
        # TODO consider resources constraint in each node, prevent resources not enough
        total_cpu_time, total_gpu_time, completion_time = self.calc_utilization(ind)
        diff_cur = max(completion_time.values()) - min(completion_time.values())
        diff_pre = 60*60*24
        max_node = max(completion_time, key=completion_time.get)
        min_node = min(completion_time, key=completion_time.get)
        while diff_cur < diff_pre:
            max_node = max(completion_time, key=completion_time.get)
            min_node = min(completion_time, key=completion_time.get)


            best_task = None
            best_diff = float('inf')

            for task in ind.task_allocation_node[max_node]:
                temp_cpu_time_max = total_cpu_time[max_node] - task['resources']['cpu'] * task['total_runtime']
                temp_gpu_time_max = total_gpu_time[max_node] - task['resources']['gpu'] * task['total_runtime']
                temp_cpu_time_min = total_cpu_time[min_node] + task['resources']['cpu'] * task['total_runtime']
                temp_gpu_time_min = total_gpu_time[min_node] + task['resources']['gpu'] * task['total_runtime']

                temp_diff_max = abs(temp_cpu_time_max - temp_gpu_time_max)
                temp_diff_min = abs(temp_cpu_time_min - temp_gpu_time_min)
                combined_diff = temp_diff_max + temp_diff_min

                if combined_diff < best_diff:
                    best_diff = combined_diff
                    best_task = task

            if best_task:
                ind.task_allocation_node[max_node].remove(best_task)
                ind.task_allocation_node[min_node].append(best_task)
                total_cpu_time[max_node] -= best_task['resources']['cpu'] * best_task['total_runtime']
                total_gpu_time[max_node] -= best_task['resources']['gpu'] * best_task['total_runtime']
                total_cpu_time[min_node] += best_task['resources']['cpu'] * best_task['total_runtime']
                total_gpu_time[min_node] += best_task['resources']['gpu'] * best_task['total_runtime']
                completion_time[max_node], _ = self.calculate_completion_time_record_with_running_task(self.node_resources[max_node], self.running_task_node[max_node], ind.task_allocation_node[max_node])
                completion_time[min_node], _ = self.calculate_completion_time_record_with_running_task(self.node_resources[min_node], self.running_task_node[min_node], ind.task_allocation_node[min_node])
                diff_pre = diff_cur
                diff_cur = max(completion_time.values()) - min(completion_time.values())
            else:
                break
            
    def check_idle_resources(self):
        pass
    
    def acquire_new_task(self,ind):
        # if resources idle, acquire new task for idle resources
        pass
                

    def calculate_completion_time_record_with_running_task(self, resources, running_task:list, task_allocation):
        current_time = time.time() # time line move
        record = current_time # start time
        ongoing_task = [] # tmp use here 
        running_seq = {}  # log seq and time information, supply for evo to choose which task to opt or mutate
        # available_resources = {node: {'cpu': res['cpu'], 'gpu': res['gpu']} for node, res in self.node_resources.items()}
        available_resources = copy.deepcopy(resources)

        # add running task
        if running_task:
            for task in running_task:
                heapq.heappush(ongoing_task, (task['finish_time'], task['resources']['cpu'], task['resources']['gpu'], task['task_id'], task['name'], task['resources']['node']))
                available_resources['cpu'] -= task['resources']['cpu']
                available_resources['gpu'] -= task['resources']['gpu']
                running_seq[task['task_id']] = {
                    'name': task['name'],
                    'task_id': task['task_id'],
                    'start_time': task['start_time'],
                    'finish_time': task['finish_time'], # refresh at finish
                    'total_runtime': task['total_runtime'], # refresh at finish
                    'resources': task['resources']
                }

        for task in task_allocation:
            # start a new task
            node = task['resources']['node']
            required_cpu = task['resources']['cpu']
            required_gpu = task['resources']['gpu']
            
            # before add each task, check if any task completed
            while ongoing_task and ongoing_task[0][0] <= current_time:
                _, cpus, gpus, finished_task_id, task_name, task_node = heapq.heappop(ongoing_task)
                available_resources['cpu'] += cpus
                available_resources['gpu'] += gpus
                task_record = running_seq[finished_task_id]
                task_record['finish_time'] = current_time
                task_record['total_runtime'] = current_time - task_record['start_time']

            # wait for task release resources
            while available_resources['cpu'] < required_cpu or available_resources['gpu'] < required_gpu:
                if ongoing_task:
                    next_finish_time, cpus, gpus, finished_task_id, task_name, task_node = heapq.heappop(ongoing_task)
                    task_record = running_seq[finished_task_id]
                    task_record['finish_time'] = current_time
                    task_record['total_runtime'] = current_time - task_record['start_time']
                    current_time = next_finish_time # time move
                    available_resources['cpu'] += cpus
                    available_resources['gpu'] += gpus
                else:
                    # all task release, resources still not enough
                    raise ValueError("Not enough resources for all tasks")

            available_resources['cpu'] -= required_cpu
            available_resources['gpu'] -= required_gpu
            start_time = current_time
            finish_time = current_time + task['total_runtime']
            heapq.heappush(ongoing_task, (finish_time, required_cpu, required_gpu, task['task_id'], task['name'], node))

            running_seq[task['task_id']] ={
                'name': task['name'],
                'task_id': task['task_id'],
                'start_time': start_time,
                'finish_time': None,
                'total_runtime': None,
                'resources': task['resources']
            }

        # time move on and log task complete
        while ongoing_task:
            next_finish_time, cpus, gpus, finished_task_id, task_name, task_node = heapq.heappop(ongoing_task)
            available_resources['cpu'] += cpus
            available_resources['gpu'] += gpus
            current_time = next_finish_time
            task_record = running_seq[finished_task_id]
            task_record['finish_time'] = current_time
            task_record['total_runtime'] = current_time - task_record['start_time']

        # 返回总完成时间和任务运行序列
        return current_time - record, running_seq
    
    def calculate_total_time(self, ind:individual):
        total_time = 0
        for task in ind.task_allocation:
            # total_time += self.hist_data.estimate_time(task)
            total_time += task['total_runtime']
        return total_time
            
    def calculate_node_info(self, ind):
        """balance each node, we should get gpu cpu consumming and total running time

        Args:
            ind (individual): _description_
        """
        pass
    
    
    def fitness(self, ind:individual, all_node=False):
        if all_node:
            # calculate total time based on avail resources and task
            total_time = 0  ## time accumulate by all task
            # completion_time = [ 0 for _ in range(len(self.node_resources.keys()))] ## HPC makespan
            completion_time = defaultdict(list)

            total_time = self.calculate_total_time(ind)
            for node in self.node_resources.keys():
                completion_time[node],ind.predict_run_seq_node[node] = self.calculate_completion_time_record_with_running_task(self.node_resources[node], self.running_task_node[node], ind.task_allocation_node[node])
            
            # ind.score = -completion_time
            ind.score = -max(completion_time.values())
            return ind.score
            
        else:
            total_time = 0
            completion_time = 0
            completion_time, ind.predict_run_seq = self.calculate_completion_time_record_with_running_task(ind.total_resources, self.running_task_node[ind.task_allocation[0]['resources']['node']], ind.task_allocation)

            ind.score = -completion_time
            return ind.score
    
    def generate_node(self):
        nodes:list= list(self.node_resources.keys())
        index = 0
        while True:
            yield nodes[index]
            index = (index + 1) % len(nodes)
    
            
    def generate_population_node(self, population_size: int):
        ## choose a node in cycle

        which_node = self.generate_node()
        ## add all task to individual
        task_nums = self.at.get_task_nums()
        all_tasks = self.at.get_all()
        population = []
        cpu_upper_bound = min(self.node_resources.values(), key=lambda x: x['cpu'])['cpu'] # node max cpu value ! consider remain resources, prevent not enough reousrces to run
        cpu_range = min(16,cpu_upper_bound) #TODO tmp test, we could simulate memory page allocate method like [1,2,4,8,16]
        gpu_upper_bound = max(self.node_resources.values(), key=lambda x: x['gpu'])['gpu']
        gpu_range = min(4,gpu_upper_bound) # should consider node GPU resources
        # TODO tmp test, resources range should determine by node
        for _ in range(population_size):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
            task_queue = self.ind_init_task_allocation() # modify to multi node, dict{'node_name':[],'node_name':[]}
            for name, ids in all_tasks.items():
                if len(ids)==0:
                    continue
                predefine_gpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'gpu')
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
        
        # initial resources minimum
        ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
        task_queue = self.ind_init_task_allocation()
        for name, ids in all_tasks.items():
            if len(ids)==0:
                continue
            predefine_gpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'gpu')
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
        
        # initial resources predifine
        ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
        task_queue = self.ind_init_task_allocation()
        for name, ids in all_tasks.items():
            if len(ids)==0:
                continue
            predefine_cpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'cpu') # this type task predifine cpu resources
            predefine_gpu = getattr(self.hist_data.queue.result_list[ids[0]].resources,'gpu')
            for task_id in ids:
                # task = self.hist_data.queue.result_list[task_id]
                # cpu = getattr(task.resources,'cpu')
                node = next(which_node)
                new_task = {
                    "name":name,
                    "task_id": task_id,
                    "resources":{
                        # "cpu": random.randint(1,16)
                        "cpu": predefine_cpu,
                        "gpu": predefine_gpu,
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
        
        # self.population = population
        return population 
    
    def generate_population_in_node(self, ind:individual, pop_size:int=10):
        """generate population in each node"""
        population_node = defaultdict(list)
        task_allocation_node = ind.task_allocation_node
        for node in self.node_resources.keys():
            task_nums = len(task_allocation_node[node])
            if task_nums == 0:
                continue
            for i in range(pop_size):
                n_ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.node_resources[node]))
                n_ind.task_allocation = copy.deepcopy(task_allocation_node[node])
                random.shuffle(n_ind.task_allocation) # shuffle task run seq
                population_node[node].append(n_ind)
            
            # use fitness function set pred run seq evo operation need it
            scores = [self.fitness(ind, all_node=False) for ind in population_node[node]]
        
        return population_node
        
    
    def mutate_cpu(self, population:list, ind_input:individual):
        ind = ind_input.copy()
        ## change resource 
        alloc = random.choice(ind.task_allocation)
        choice = [-5,-3,-2,-1,0,1,2,3,5]
        new_alloc = alloc['resources']['cpu'] + random.choice(choice)
        
        if new_alloc <= 0:
            alloc['resources']['cpu'] = 1
        elif new_alloc >= self.node_resources[alloc['resources']['node']]['cpu']:
            new_alloc = self.node_resources[alloc['resources']['node']]['cpu']
        else:
            alloc['resources']['cpu'] = new_alloc
        
        population.append(ind)
    
    def mutate_seq(self,population:list, ind_input:individual):
        ind = ind_input.copy()
        ## change task sequence
        index1 = random.randrange(len(ind.task_allocation))
        index2 = random.randrange(len(ind.task_allocation))
        while index2 == index1:
            index2 = random.randrange(len(ind.task_allocation))
            
        ind.task_allocation[index1], ind.task_allocation[index2] = ind.task_allocation[index2], ind.task_allocation[index1]
        
        population.append(ind)
    
    
    def crossover_arith_ave(self, population:list, ind_input1:individual, ind_input2:individual):
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
            task['resources']['cpu'] = (ind_input1.get_task_resources(name,task_id)['cpu']+ind_input2.get_task_resources(name,task_id)['cpu'])//2
        population.append(ind1)
    
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

    def crossover_pmx(self, population:list, ind_input1:individual, ind_input2:individual):
        ind1 = ind_input1.copy()
        ind2 = ind_input2.copy()
        size = len(ind1.task_allocation)
        p1, p2 = [0]*size, [0]*size
        
        cxpoint1 = random.randint(0, size-1)
        cxpoint2 = random.randint(0, size-1)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
        # print(cxpoint1,cxpoint2)
        for i in range(cxpoint1,cxpoint2+1):
            p1[i] = ind2.task_allocation[i]
            p2[i] = ind1.task_allocation[i]
            
        for i in range(size):
            if i < cxpoint1 or i > cxpoint2:
                ii = ind1.task_allocation[i]
                while self.list_dict_found(p1[cxpoint1:cxpoint2+1],ii):
                    # ii = ind1.task_allocation[p1[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind1.task_allocation[self.list_dict_index(ind2.task_allocation,ii)]
                p1[i] = ii
                
                ii = ind2.task_allocation[i]
                while self.list_dict_found(p2[cxpoint1:cxpoint2+1],ii):
                    # ii = ind2.task_allocation[p2[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind2.task_allocation[self.list_dict_index(ind1.task_allocation,ii)]
                p2[i] = ii
        
        ind1.task_allocation = p1
        ind2.task_allocation = p2
        population.append(ind1)
        population.append(ind2)
        
    def opt1(self, population:list, ind:individual):
        ind = ind.copy()
        ## add resources for longest task
        # logger.info(f"opt1: {ind.task_allocation}")
        # logger.info(f"opt1: {ind.predict_run_seq}")
        tasks = sorted(ind.predict_run_seq.items(),key=lambda x:x[1]['total_runtime'], reverse=True) # sorted dict return desending list(tuple) 
        for i in range(len(tasks)//3): # long 1/3 percent
            index = self.list_dict_index(ind.task_allocation,tasks[i][1])
            # task may in running, so we need to check if it is in the task allocation
            if index:
                new_alloc = random.choice([1,2,3,4,5]) + ind.task_allocation[index]['resources']['cpu']
                if new_alloc <= max(self.node_resources.values(), key=lambda x: x['cpu'])['cpu']//2: # only allow at constrait resources
                    ind.task_allocation[index]['resources']['cpu'] = new_alloc
                
        ## remove resources for shortest task
        # task = min(ind.predict_run_seq, key=lambda x:x['total_runtime'])
            index = self.list_dict_index(ind.task_allocation,tasks[-i][1]) # short 1/3 percent
            # task may in running, so we need to check if it is in the task allocation
            if index:
                # for caution, we jsut minus 1
                new_alloc = ind.task_allocation[index]['resources']['cpu'] - 1
                if new_alloc >= 1:
                    ind.task_allocation[index]['resources']['cpu'] = new_alloc
                    
        population.append(ind)
        
    def opt2(self, population:list, ind:individual):
        ind = ind.copy()
        ## advance the latest task order
        tasks = sorted(ind.predict_run_seq.items(), key=lambda x:x[1]['finish_time']) # asending
        task = tasks[-1][1]
        index = self.list_dict_index(ind.task_allocation,task)
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
            
    def process_individual_mutate(self, population, ind1,ind2,crossover_rate=0):
        self.mutate_cpu(population, ind1)
        self.mutate_cpu(population, ind2)

        self.mutate_seq(population, ind1)
        self.mutate_seq(population, ind2)
        
        self.crossover_pmx(population, ind1,ind2)
        self.crossover_arith_ave(population, ind1,ind2)
        
    def check_generation_node(self, population):
        logger_buffer = {}
        for ind in population:
            for node in ind.task_allocation_node.keys():
                if len(ind.task_allocation_node[node]) == 0:
                    logger_buffer[ind.individual_id] = f"Node {node} has no task"
    
    def clean_population(self, population):
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 ** 2  # in MB
        print(f"Current memory usage: {memory_usage:.2f} MB")
        for ind in population:
            del ind
        gc.collect()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 ** 2  # in MB
        print(f"After cleaning memory usage: {memory_usage:.2f} MB")
            
    # we do not extend copied ind for evolution, we copy evoluting ind
    def run_ga(self, num_population:int=10, num_generations_all:int=5, num_generations_node:int=20):
        logger.info(f"Starting GA with available tasks: {self.at.get_all()}")
        logger.info(f"on going task:{self.running_task_node}, resources:{self.resources}, node_resources:{self.node_resources}")
        def safe_execute(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception occurred: {e}", exc_info=True)
                return None
        # TODO try multi process here, check if memory leak
        # run no record task
        ind = self.detect_no_his_task()
        if len(ind.task_allocation) > 0:
            return ind
        
        self.population = self.generate_population_node(1)
        
        # train random forest model for predict running time
        self.hist_data.random_forest_train()

        
        pop_size = len(self.population)

        # first we can estimate every task running time
        self.hist_data.estimate_batch(self.population, all_node=True)
        scores = [self.fitness(ind, all_node=True) for ind in self.population]
        self.population = [self.population[i] for i in np.argsort(scores)[::-1]]
        # only one task , submit directly
        if self.at.get_total_nums() == 1:
            logger.info(f"Only one task, submit directly")
            return self.population[0]
        # logger.info(f"Generation 0: {population[0]}")
        score = self.population[0].score
        logger.info(f"initial score is {score}")
        new_score = 0
        # for gen in range(4):
        #     # load balance on each node
        #     # TODO multinode pmx and mutate
        #     # TODO load balancing here
            
        #     # evo on each node
        #     for a_ind in self.population:
        #         self.load_balance(a_ind)
        #         self.population_node = self.generate_population_in_node(a_ind,20) # regenerate
        #         # logger.info(f"Generation_node, gen: {gen}: a_ind:{a_ind}")
        #         for node in self.node_resources.keys():
        #             logger.info(f"Node {node} evolution")
        #             population = self.population_node[node] # must shallow copy here
        #             for gen_node in range(20):
        #                 population = population[:20] # control nums
        #                 # logger.info(f"show population{population}")
        #                 random.shuffle(population)
        #                 with concurrent.futures.ThreadPoolExecutor() as executor:
        #                     futures = []
        #                     size = len(population)
        #                     # logger.info(f"size:{size}")
        #                     for i in range(size // 2):
        #                         ind1 = population[i]
        #                         ind2 = population[size - i - 1]
        #                         if new_score == score:  # no change, add stimulation
        #                             futures.append(executor.submit(safe_execute, self.process_individual_mutate, population, ind1, ind2))
        #                         futures.append(executor.submit(safe_execute, self.opt1, population, ind1))
        #                         futures.append(executor.submit(safe_execute, self.opt1, population, ind2))
        #                         futures.append(executor.submit(safe_execute, self.opt2, population, ind1))
        #                         futures.append(executor.submit(safe_execute, self.opt2, population, ind2))
        #                     concurrent.futures.wait(futures)
        #                 # size = len(population)
        #                 # # logger.info(f"size:{size}")
        #                 # for i in range(size // 2):
        #                 #     ind1 = population[i]
        #                 #     ind2 = population[size - i - 1]
        #                 #     if new_score == score:  # no change, add stimulation
        #                 #         self.process_individual_mutate(population, ind1, ind2)
        #                 #     self.opt1(population, ind1)
        #                 #     self.opt1(population, ind2)
        #                 #     self.opt2(population, ind1)
        #                 #     self.opt2(population, ind2)
        #                 self.hist_data.estimate_batch(population, all_node=False) # cal predict running time for each task
        #                 scores = [self.fitness(ind, all_node=False) for ind in population]
        #                 population = [population[i] for i in np.argsort(scores)[::-1]]
        #                 score = new_score
        #                 new_score = population[0].score
        #                 population = population[:50]
        #             logger.info(f"Generation {gen}-{gen_node}: best ind on node{node} score:{population[0].score}")
        #             # best ind on node
        #             # logger.info(f"Generation_node {gen_node}: best ind on node{node}:{population[0]}")
        #             best_ind = max(population, key=lambda ind: ind.score)
        #             a_ind.task_allocation_node[node]=copy.deepcopy(best_ind.task_allocation)
        #             # cleanning
        #             self.clean_population(population)
                    
        #             # single node ind operation end
        #         # all node ind operation end
        #     # global int operation here
        #     scores = [self.fitness(ind, all_node=True) for ind in self.population]
        #     logger.info(f"Generation {gen}: best ind score:{self.population[0].score}")
            
        # best ind global
        best_ind = max(self.population, key=lambda ind: ind.score)
        logger.info(f"score of all ind:{scores}")
        logger.info(f"Best ind:{best_ind}")
        self.best_ind = best_ind
        best_allocation = []
        for key in best_ind.task_allocation_node.keys():
            best_allocation.extend(best_ind.task_allocation_node[key])
        return best_allocation
    

########## for test
class test_colmena_queue():
    def __init__(self):
        self.result_list = {}


if __name__ == "__main__":
    
    
    
    hist_path = [] 
    hist_path.append("/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/simulation-results-20240319_230707.json")
    hist_path.append("/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/inference-results-20240319_230707.json")
    hist_path.append("/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/sampling-results-20240319_230707.json")
    hist_path.append("/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/training-results-20240319_230707.json")
    
    methods = ['run_calculator', 'run_sampling', 'train', 'evaluate']
    topics = ['simulate', 'sample', 'train', 'infer']
    test_queue = test_colmena_queue()
    my_available_task = {}
    for method in methods:
        my_available_task[method] = []
    
    with open("/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/task_queue_audit.pkl",'rb')as f:
        hist_task_queue_audit = pickle.load(f)
    
    evosch = evosch2(resources={"node1":{"cpu":56,"gpu":4},"node2":{"cpu":56,"gpu":4}}, at=available_task(my_available_task), hist_data=historical_data(methods=methods, queue=test_queue))
    
    evosch.hist_data.get_features_from_his_json(hist_path)
    
    # add data class
    from fff.simulation.utils import read_from_string, write_to_string
    # add task
    for _ in range(10):
        to_run_f = self.hist_task_queue_audit.pop(0)
        task_type = 'audit'
        atoms = to_run_f.atoms
        atoms.set_center_of_mass([0, 0, 0])
        xyz = write_to_string(atoms, 'xyz')
        
        result = Result(
            (input_args, input_kwargs),
            method=method,
            topic=topic,
            keep_inputs=_keep_inputs,
            serialization_method=self.serialization_method,
            task_info=task_info,
            # Takes either the user specified or a default
            resources=resources or ResourceRequirements(),
            **ps_kwargs
        )
        result.time_serialize_inputs, proxies = result.serialize()
    
    
        
        