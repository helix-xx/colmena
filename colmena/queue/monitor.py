from dataclasses import dataclass, field  # 用于定义 `@dataclass` 和 `field`
from typing import List, Dict, Union, Any, Collection  # 用于类型注解
import threading  # 用于线程锁 `threading.Lock`
import copy  # 用于深拷贝 `copy.deepcopy`
import json  # 用于 JSON 文件的解析 `json.loads`
from colmena.models import Result
import logging  # 用于日志记录 `logger.warning`
import uuid

from .predictor import TaskTimePredictor 

logger = logging.getLogger(__name__)

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonClass(metaclass=SingletonMeta):
    def __init__(self):
        pass


class Sch_data(SingletonClass):
    def __init__(self, methods, available_resources):
        self.result_list = {}
        self.sch_task_list = {}
        self.pilot_task = {}
        self.Task_time_predictor: TaskTimePredictor = None
        self.avail_task: available_task = None
        self.avail_task_cap: int = None
        self.methods = methods
        self.available_resources = available_resources

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