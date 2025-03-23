import os  # 用于文件路径操作
import time  # 用于时间测量和延迟操作
import copy  # 用于深拷贝对象
import threading  # 用于线程和线程锁
import multiprocessing  # 用于多进程池
import logging  # 用于日志记录
import uuid  # 用于生成唯一任务ID
import numpy as np  # 用于数值计算
from typing import Callable, Optional
import concurrent.futures

logger = logging.getLogger(__name__)

from colmena.models import Result
from .evo_sch import evosch2, individual, precalculate_fixed_state
from .fcfs_sch import FCFSScheduler
from .monitor import available_task, HistoricalData, Sch_data
from .mrsa_sch import run_mrsa_scheduler


class SchedulerTimer:
    """Manages scheduling triggers with intelligent timing based on task execution windows"""
    
    def __init__(self, trigger_callback: Callable, scheduling_time: int = 120):
        """
        Args:
            trigger_callback: Callback function to execute when timer triggers
            scheduling_time: Time needed for one scheduling cycle in seconds
        """
        self.trigger_callback = trigger_callback
        self.scheduling_time = scheduling_time
        self.min_trigger_time = 10
        self.timer: Optional[threading.Timer] = None
        self.timer_lock = threading.Lock()
        self._timeout = 0
        
    def _execute_callback(self):
        """Wrapper to execute the callback and clear timer reference"""
        try:
            self.trigger_callback()
        finally:
            with self.timer_lock:
                self.timer = None

    def reset(self, task_lists: np.ndarray):
        """Reset the scheduling timer based on running tasks and scheduling window
        
        Args:
            task_lists: Numpy structured array of tasks with start_time and finish_time fields
        """
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None

            # Default timeout is scheduling_time
            timeout = self.min_trigger_time
            
            if task_lists is not None and len(task_lists) > 0:
                current_time = time.time()
                
                # Get unique nodes
                unique_nodes = np.unique(task_lists['node'])
                
                # Find last task for each node
                node_last_tasks = {}
                for node in unique_nodes:
                    # Get tasks for this node
                    node_tasks = task_lists[task_lists['node'] == node]
                    if len(node_tasks) > 0:
                        # Sort by start_time and get the last one
                        sorted_indices = np.argsort(node_tasks['start_time'])
                        last_task = node_tasks[sorted_indices[-1]]
                        node_last_tasks[node] = last_task
                        logger.debug(f"Node {node} last task {last_task['task_id']} "
                                f"starts at {last_task['start_time']}")

                if node_last_tasks:
                    # Find the earliest start time among last tasks
                    earliest_start = float('inf')
                    # earliest_node = None
                    # earliest_task = None
                    
                    for node, task in node_last_tasks.items():
                        if task['start_time'] < earliest_start:
                            earliest_start = task['start_time']
                            earliest_node = node
                            earliest_task = task

                    # Calculate timeout based on earliest start time
                    time_until_start = earliest_start - current_time
                    
                    if time_until_start > self.scheduling_time:
                        # Set timer to trigger before the earliest last task starts
                        # timeout = time_until_start - self.scheduling_time
                        timeout = self.scheduling_time
                        logger.info(f"Setting timer for {timeout}s before last task "
                                f"{earliest_task['task_id']} on node {earliest_node} "
                                f"(starts at {earliest_start})")
                    else:
                        # If the earliest start time is too close or in the past,
                        # use default scheduling time
                        timeout = time_until_start
                        logger.info(f"Earliest last task starts too soon, "
                                f"using default timeout of {timeout}s")

            # Create and start new timer
            self._time_out = timeout
            self.timer = threading.Timer(timeout, self._execute_callback)
            self.timer.start()
            logger.info(f"Reset scheduling timer for {timeout} seconds")

    def cancel(self):
        """Cancel current timer if exists"""
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
                
class FeedbackEvent():
    def __init__(self, methods):
        self.methods = methods
        self.feedback_events = {}
        self.feedback_info = {}
        for method in methods:
            self.feedback_events[method] = threading.Event()
            
    def set_event(self):
        pass
    
    def get_event(self):
        pass

    
                
## resources checking and events handling
                
class SmartScheduler:
    # support different scheduling policy here
    
    ## init all sch model here
    # sch_data can be menber of all member model
    def __init__(self, methods, available_task_capacity, available_resources, sch_config= None, scheduling_time:int=120, scheduler_type="ga"):
        self.sch_data: Sch_data = Sch_data(methods, available_resources, scheduler_type)
        # self.agent_pilot = agent_pilot(sch_data=self.sch_data, resources_rate=2, available_resources=available_resources, util_level=0.8)
        self.sch_data.init_task_queue(available_task(methods), available_task_capacity)
        self.sch_data.init_hist_task(HistoricalData(methods))
        self.sch_data.init_task_time_predictor(methods, self.sch_data.historical_task_data.features)
        self.evo_sch: evosch2 = evosch2(resources=available_resources, at=self.sch_data.avail_task, hist_data=self.sch_data.historical_task_data, sch_data=self.sch_data)
        self.fcfs_sch: FCFSScheduler = FCFSScheduler(resources=available_resources, at=self.sch_data.avail_task, hist_data=self.sch_data.historical_task_data, sch_data=self.sch_data)
        self.feedback_event: FeedbackEvent = FeedbackEvent(methods)
        
        
        #agent_pilot
        self.resources_rate = 2
        self.available_resources = available_resources
        self.record_resources = copy.deepcopy(available_resources)
        self.util_level = 0.1
        self.exceed_area_limit = 1.1
        self.exceed_completion_time_limit = 1
        
        # scheduler timer
        # self.scheduler_timer:SchedulerTimer = None
        self._scheduling_time = scheduling_time
        # scheduler result_ind, allocation in available task class
        self.best_result = None
        
        # lock
        self.sch_lock = threading.Lock()
        self.available_task_lock = threading.Lock() # lock for available task to move task between available and scheduled
        
        # processes = len(self.node_resources)
        processes = 16
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
        # hist_path.append(
        #     os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/qimingdata/simulation-results-more_core.json')
        # )
        hist_path.append(
            os.path.join(self.sch_data.usr_path, 'project/colmena/multisite_/finetuning-surrogates/runs/hist_data/qimingdata/training-results_v100.json')
        )
        self.sch_data.historical_task_data.get_features_from_his_json(hist_path)
        self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data)
        self.sch_data.Task_time_predictor.fill_time_running_database(self.available_resources, self.sch_data.historical_task_data.historical_data)
        # self.sch_data.Task_time_predictor.fill_features_from_new_task(self.available_resources, self.sch_data.sch_task_list)
        self.sch_data.Task_time_predictor.fill_runtime_records_with_predictor()
        
        logger.info('init smart scheduler')
        
        # warmup numba functions
        # self._warmup_numba_functions()
        
    def __del__(self):
        self.pool.close()
        self.pool.join()
        
    def set_scheduler_timer(self, trigger_callback: Callable):
        """设置调度定时器的回调函数
        
        Args:
            trigger_callback: 触发调度的回调函数
        """
        self.scheduler_timer = SchedulerTimer(
            trigger_callback=trigger_callback,
            scheduling_time=self._scheduling_time
        )
        
    # def acquire_resources(self, key):
    #     # topic与method不一样，暂时添加一个映射
    #     topic_method_mapping = {
    #         'simulate': 'run_calculator',
    #         'sample': 'run_sampling',
    #         'train': 'train',
    #         'infer': 'evaluate'
    #     }
        
    #     method = topic_method_mapping.get(key, None)  # 根据给定的 key 获取对应的方法
        
    #     pilot_task = self.sch_data.pilot_task.get(method, None)
    #     info = {}
    #     if not pilot_task:
    #         info['reason'] = "no pilot task"
    #         return 0, info
    #     else:
    #         # 获取前后分配的情况，并通过预测器的时间计算资源利用率
    #         ind = self.best_result
    #         if not ind:
    #             info['reason'] = 'no previous info'
    #             return 0, info
    #         total_cpu_time_per_node, total_gpu_time_per_node, completion_time = self.evo_sch.calc_utilization(ind)
            
    #         all_tasks = self.sch_data.avail_task.get_all()
    #         all_tasks = copy.deepcopy(all_tasks)
    #         pilot_task = copy.deepcopy(pilot_task)
    #         self.sch_data.renew_task_uuid(pilot_task)
            
    #         all_tasks = self.sch_data.avail_task.dummy_add_task_id(method, pilot_task['task_id'], all_tasks=all_tasks)
    #         self.sch_data.add_sch_task(pilot_task)
    #         _ = self.run_sch()
    #         new_ind = self.sch_data.best_ind
    #         new_total_cpu_time_per_node, new_total_gpu_time_per_node, new_completion_time = self.evo_sch.calc_utilization(new_ind)
    #         self.sch_data.pop_sch_task(pilot_task)
            
    #         # evaluate resources
    #         node_cpu_count = {node: self.available_resources[node]['cpu'] for node in self.available_resources.keys()}
    #         node_gpu_count = {node: self.available_resources[node]['gpu'] for node in self.available_resources.keys()}

    #         current_cpu_utilization = {node: (total_cpu_time_per_node[node] / (completion_time[node] * node_cpu_count[node]) if completion_time[node] > 0 else 0) for node in total_cpu_time_per_node}
    #         new_cpu_utilization = {node: (new_total_cpu_time_per_node[node] / (new_completion_time[node] * node_cpu_count[node]) if new_completion_time[node] > 0 else 0) for node in new_total_cpu_time_per_node}
            
    #         current_gpu_utilization = {node: (total_gpu_time_per_node[node] / (completion_time[node] * node_gpu_count[node]) if completion_time[node] > 0 else 0) for node in total_gpu_time_per_node}
    #         new_gpu_utilization = {node: (new_total_gpu_time_per_node[node] / (new_completion_time[node] * node_gpu_count[node]) if new_completion_time[node] > 0 else 0) for node in new_total_gpu_time_per_node}
            
    #         # 计算利用率提升比例
    #         utilization_improvement = {}
    #         for node in current_cpu_utilization:
    #             cpu_improvement_ratio = (
    #                 (new_cpu_utilization[node] - current_cpu_utilization[node]) / current_cpu_utilization[node]
    #                 if current_cpu_utilization[node] > 0 else float('inf')
    #             )
    #             gpu_improvement_ratio = (
    #                 (new_gpu_utilization[node] - current_gpu_utilization[node]) / current_gpu_utilization[node]
    #                 if current_gpu_utilization[node] > 0 else float('inf')
    #             )
                
    #             utilization_improvement[node] = {
        
    #                 'cpu': cpu_improvement_ratio,
    #                 'gpu': gpu_improvement_ratio
    #             }
                
    #         # 计算计算时间的延长
    #         completion_time_improvement = {node: (new_completion_time[node] - completion_time[node]) / completion_time[node]  if completion_time[node] > 0 else 0 for node in completion_time}
            
    #         used_cpu_area = sum(total_cpu_time_per_node.values())
    #         new_used_cpu_area = sum(new_total_cpu_time_per_node.values())
    #         used_gpu_area = sum(total_gpu_time_per_node.values())
    #         new_used_gpu_area = sum(new_total_gpu_time_per_node.values())
            
    #         cpu_area_per_node = {}
    #         new_cpu_area_per_node = {}
    #         gpu_area_per_node = {}
    #         new_gpu_area_per_node = {}

    #         # completion time with cpu and gpu weight
    #         for node in completion_time:
    #             cpu_nums = self.available_resources[node]['cpu']
    #             cpu_area_per_node[node] = completion_time[node] * cpu_nums
    #             new_cpu_area_per_node[node] = new_completion_time[node] * cpu_nums
    #             gpu_nums = self.available_resources[node]['gpu']
    #             gpu_area_per_node[node] = completion_time[node] * gpu_nums
    #             new_gpu_area_per_node[node] = new_completion_time[node] * gpu_nums
            
    #         total_cpu_area = sum(cpu_area_per_node.values())
    #         new_total_cpu_area = sum(new_cpu_area_per_node.values())
    #         total_gpu_area = sum(gpu_area_per_node.values())
    #         new_total_gpu_area = sum(new_gpu_area_per_node.values())
            
    #         # total util improvement
    #         # TODO area 可能不变，用标准任务 / area获得潜在效率提升
    #         # 检查占用总area是否超出：
    #         if new_total_cpu_area > self.exceed_area_limit * total_cpu_area or new_total_gpu_area > self.exceed_area_limit * total_gpu_area:
    #             info['reason'] = "exceed area limit"
    #             return 0, info
    #         # 检查最大 completion time是否超出
    #         if max(new_completion_time.values()) > self.exceed_completion_time_limit * max(completion_time.values()):
    #             info['reason'] = "exceed completion time limit"
    #             return 0, info
            
    #         # 检查是否有提升达到设定的 util_level
    #         # 设置的提升阈值（例如10%）
    #         info = utilization_improvement
    #         for node, improvement in utilization_improvement.items():
    #             if improvement['cpu'] >= self.util_level or improvement['gpu'] >= self.util_level:
    #                 info['reason'] = "utilize improvement;"
    #                 return 1, info
            
    #         # info['reason'] = "no utilize improvement"
    #         # logger.info('acquire resources info {}'.format(info))
    #         # return 0, info
    #         info['reason'] = "no limit exceed"
    #         return 1, info
        
    def acquire_resources(smart_scheduler, key):
        # 配置参数
        UTIL_LOW_THRESHOLD = 0.9  # 资源利用率低水位阈值
        TIME_LIMIT = 0.2          # 允许完成时间最大增幅
        
        # topic与method不一样，暂时添加一个映射
        topic_method_mapping = {
            'simulate': 'run_calculator',
            'sample': 'run_sampling',
            'train': 'train',
            'infer': 'evaluate'
        }
        
        method = topic_method_mapping.get(key, None)  # 根据给定的 key 获取对应的方法
        
        pilot_task = smart_scheduler.sch_data.pilot_task.get(method, None)
        info = {}
        
        base_ind = smart_scheduler.best_result
        if not pilot_task:
            info['reason'] = "no pilot task"
            return 0, info

        # 获取前后分配的情况，并通过预测器的时间计算资源利用率
        if not base_ind:
            info['reason'] = 'no previous info'
            return 0, info
        base_cpu_area, base_gpu_area, base_completion, _ = smart_scheduler.evo_sch.calc_utilization(base_ind)
        base_max_time = max(base_completion.values()) if base_completion else 0
        
        # run_sch again
        # all_tasks = self.sch_data.avail_task.get_all()
        # all_tasks = copy.deepcopy(all_tasks)
        # pilot_task = copy.deepcopy(pilot_task)
        # self.sch_data.renew_task_uuid(pilot_task)
        
        # all_tasks = self.sch_data.avail_task.dummy_add_task_id(method, pilot_task['task_id'], all_tasks=all_tasks)
        # self.sch_data.add_sch_task(pilot_task)
        # _ = self.run_sch()
        # new_ind = self.sch_data.best_ind
        # new_total_cpu_time_per_node, new_total_gpu_time_per_node, new_completion_time = self.evo_sch.calc_utilization(new_ind)
        # self.sch_data.pop_sch_task(pilot_task)
        
        # direct get data
        new_task = copy.deepcopy(pilot_task)
        smart_scheduler.sch_data.renew_task_uuid(new_task)
        new_ind = individual(tasks_nums=base_ind.tasks_nums+1, total_resources=base_ind.total_resources)
        new_ind.task_array[:-1] = base_ind.task_array
        
        cpu = new_task['resources.cpu']
        gpu = new_task['resources.gpu']
        msg_size = new_task['message_sizes.inputs']
        method = new_task['method']
        task_id = new_task['task_id']
        
        new_ind.task_array[-1] = (
            method,
            task_id,
            cpu,
            gpu,
            "node",
            0,
            0,
            0,
        )
    
        for node, resources in base_ind.total_resources.items():
            if resources['cpu'] < cpu or resources['gpu'] < gpu:
                continue
            new_ind.task_array[-1]['node'] = node
            # TODO how to choose best resources
            # new_ind.task_array[-1]['cpu'] = resources['cpu']
            # new_ind.task_array[-1]['gpu'] = resources['gpu']
            new_ind.task_array[-1]['total_runtime'] = smart_scheduler.sch_data.Task_time_predictor.get_runtime(cpu, gpu, msg_size, method)
            new_ind.update_task_id_index()
            
            # TODO 此处可优化，仅需计算节点上的任务更改
            new_cpu_area, new_gpu_area, new_completion, _ = smart_scheduler.evo_sch.calc_utilization(new_ind)
            new_max_time = max(new_completion.values()) if new_completion else 0
        #     print(
        #         f"Node: {node}, "
        #         f"New CPU Time: {new_total_cpu_time_per_node[node]}, "
        #         f"Total CPU Time: {total_cpu_time_per_node[node]}"
        #     )
        #     print(
        #         f"Node: {node}, "
        #         f"New GPU Time: {new_total_gpu_time_per_node[node]}, "
        #         f"Total GPU Time: {total_gpu_time_per_node[node]}"
        #     )
        #     print(
        #         f"Node: {node}, "
        #         f"New Completion Time: {new_completion_time[node]}, "
        #         f"Completion Time: {completion_time[node]}"
        #     )
        #     print(
        #         f"Node: {node}, "
        #         f"New Runtime: {new_total_runtime[node]}, "
        #         f"Total Runtime: {total_runtime[node]}"
        #     )
        # print(new_ind.task_array)
            # 如果新任务不会延长completion time
            if new_max_time <= base_max_time:
                return 1, info 
            # 如果新任务可以解决资源失配 #TODO 且不能无限制提交任务
            # 计算节点原始利用率
            # base_node_time = base_completion.get(node, 0)
            # base_cpu_util = (base_cpu_area[node] / (base_max_time * resources['cpu'])) if base_max_time > 0 else 0
            # base_gpu_util = (base_gpu_area[node] / (base_max_time * resources['cpu'])) if base_max_time > 0 else 0
            
            # # 计算新利用率
            # new_cpu_util = (new_cpu_area[node] / (new_max_time * resources['cpu'])) if new_max_time > 0 else 0
            # new_gpu_util = (new_gpu_area[node] / (new_max_time * resources['gpu'])) if new_max_time > 0 else 0
            
            # # 时间的延长
            # time_increase_ratio = (new_max_time - base_max_time) / base_max_time if base_max_time > 0 else 0
            
            # util_improve = new_cpu_util > base_cpu_util and new_gpu_util > base_gpu_util
            # # 资源类型独立判断
            # cpu_improved = base_cpu_util < UTIL_LOW_THRESHOLD and new_cpu_util > base_cpu_util
            # gpu_improved = base_gpu_util < UTIL_LOW_THRESHOLD and new_gpu_util > base_gpu_util
            
            # # 满足任一资源类型改进且时间可控
            # if util_improve and (cpu_improved or gpu_improved) and time_increase_ratio <= TIME_LIMIT:
            #     improvements = []
            #     if cpu_improved:
            #         improvements.append(f"CPU+{(new_cpu_util - base_cpu_util):.1%}")
            #     if gpu_improved:
            #         improvements.append(f"GPU+{(new_gpu_util - base_gpu_util):.1%}")
            #     return 1, {
            #         'reason': f'util improved ({", ".join(improvements)}) at {node}',
            #         'node': node
            #     }
        
        return 0, {'reason': 'no suitable condition met'}
        
    def check_runtime_resources(self):
        # 在进行完调度后计算资源失配情况
        
        # 获取调度的最佳结果
        best_ind = self.sch_data.best_ind
        
    def get_feedback_event():
        # 等待事件并获取判断资源是否失配的信息
        pass
    
    def check_resources_and_wait(topic, agent_events):
        # 同时等待获取资源信息的事件和agent设置的事件
        pass

    def _evaluate_resources_for_all_agents(self):
        """
        异步评估所有agent类型的资源状态，并设置相应的事件
        """
        # 确保resource_events字典已初始化
        if not hasattr(self, 'resource_feedback_events'):
            self.resource_feedback_events = {}
            self.resource_feedback_info = {}
        
        # 获取所有任务类型
        task_types = ['simulate', 'sample', 'train', 'infer']
        
        # 为每种任务类型评估资源状态
        for task_type in task_types:
            try:
                permit, info = self.acquire_resources(task_type)
                topic_method_mapping = {
                    'simulate': 'run_calculator',
                    'sample': 'run_sampling',
                    'train': 'train',
                    'infer': 'evaluate'
                }
                method = topic_method_mapping.get(task_type, None)
                
                # 确保该方法有对应的事件
                if method not in self.resource_feedback_events:
                    self.resource_feedback_events[method] = threading.Event()
                    self.resource_feedback_info[method] = {'permit': 0, 'reason': 'not evaluated yet'}
                
                # 根据评估结果设置或清除事件
                if permit == 1:
                    self.resource_feedback_events[method].set()
                    self.resource_feedback_info[method] = info
                    self.resource_feedback_info[method]['permit'] = 1
                    logger.info(f"Setting resource event for {task_type} - additional tasks can be submitted")
                else:
                    self.resource_feedback_events[method].clear()
                    self.resource_feedback_info[method] = info
                    self.resource_feedback_info[method]['permit'] = 0
                    logger.info(f"Clearing resource event for {task_type} - no additional tasks should be submitte, info{info}")
            except Exception as e:
                logger.error(f"Error evaluating resources for {task_type}: {e}")
            
    def run_sch(self, method = "ga", model_type="powSum", scheduler_time=time.time()):
        """运行调度器

        Args:
            method (str, optional): 选择ga或mrsa调度器. Defaults to "ga".
            model_type (str, optional): 选择mrsa时设置amdMax, amdSum, powMax, powSum四种性能模型. Defaults to "powSum".

        Returns:
            _type_: _description_
        """
        # run evo sch
        # with self.sch_lock: # 异步进行不能在这里加锁，每个调度算法都有自己的可调度任务 目前这里没有考虑异步的情况是否正常运行
        # init / fill task database
        # self.sch_data.Task_time_predictor.train(self.sch_data.historical_task_data.historical_data) # 可开启每次调度时训练一次模型
        self.sch_data.Task_time_predictor.fill_features_from_new_task(self.available_resources, self.sch_data.sch_task_list)
        self.sch_data.Task_time_predictor.fill_runtime_records_with_predictor()
        logger.info(f"available task:{self.sch_data.avail_task.task_ids}, scheduled task: {self.sch_data.avail_task.scheduled_task}")
        all_tasks, scheduled_array = self.sch_data.avail_task.get_schedulable_tasks(self.scheduler_timer.scheduling_time, scheduler_time)
        self.sch_data.avail_task.move_available_to_scheduled(all_tasks)
        
        precalculate_fixed_state(self.sch_data, self.sch_data.running_task_node, self.sch_data.avail_task.allocations, scheduler_time=scheduler_time)
        if method == "ga":
            best_allocation = self.evo_sch.run_ga_v2(all_tasks, pool = self.pool)
            self.best_result = self.sch_data.best_ind
            self._evaluate_resources_for_all_agents() # 通过反馈 动态调整任务负载
        elif method == "mrsa":
            best_allocation = run_mrsa_scheduler(sch_data=self.sch_data, model_type=model_type, tasks=all_tasks) # not with start time finish time infomation
            mrsa_ind = self.sch_data.best_ind
            self.evo_sch.fitness(mrsa_ind)
            best_allocation = mrsa_ind.task_array
            
        self.sch_data.avail_task.move_allocation_to_scheduled(best_allocation) # 线程安全
        
        return best_allocation