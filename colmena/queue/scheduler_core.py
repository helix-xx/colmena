import os  # 用于文件路径操作
import time  # 用于时间测量和延迟操作
import copy  # 用于深拷贝对象
import threading  # 用于线程和线程锁
import multiprocessing  # 用于多进程池
import logging  # 用于日志记录
import uuid  # 用于生成唯一任务ID
import numpy as np  # 用于数值计算

logger = logging.getLogger(__name__)

from colmena.models import Result
from .evo_sch import evosch2, individual
from .fcfs_sch import FCFSScheduler
from .monitor import available_task, HistoricalData, Sch_data


    
    
class SmartScheduler:
    # support different scheduling policy here
    
    ## init all sch model here
    # sch_data can be menber of all member model
    def __init__(self, methods, available_task_capacity, available_resources, sch_config= None):
        self.sch_data: Sch_data = Sch_data(methods, available_resources)
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

        hist_path.append(
            os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/inference-results-20240319_230707.json')
        )
        hist_path.append(
            os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/sampling-results-20241211.json')
        )
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
        # _calculate_completion_time(
        #     small_task_cpu,
        #     small_task_gpu,
        #     small_task_runtime,
        #     empty_running,
        #     empty_cpus,
        #     empty_gpus,
        #     4,
        #     2,
        #     0.0
        # )
        # print(f"Warmed up in {time.time() - start:.2f} seconds")
        
    
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