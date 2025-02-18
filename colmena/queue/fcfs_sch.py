from collections import defaultdict
import threading
import copy
import logging
from colmena.models import Result
from .monitor import available_task, HistoricalData, Sch_data

logger = logging.getLogger(__name__)

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