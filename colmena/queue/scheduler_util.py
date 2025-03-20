from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

task_dtype  = [
            ('name', 'U20'), 
            ('task_id', 'U40'),
            ('cpu', 'i4'),
            ('gpu', 'i4'),
            ('node', 'U10'),
            ('total_runtime', 'f8'),
            ('cpu_area', 'f8'), # tmp use for area
            ('gpu_area', 'f8'),
            ('feature_msg_size', 'f8')
        ]

def distribute_tasks(tasks, nodes, sch_task_lists, Task_time_predictor):
    """简单通过基准资源消耗负载均衡任务到多个节点上"""
    node_load = defaultdict(lambda:{'cpu':0.0, 'gpu':0.0})

    task_nums = sum(len(ids) for ids in tasks.values())
    task_details = np.zeros(task_nums, dtype=task_dtype)
    task_cnt = 0
    for name, ids in tasks.items():
        if not ids:
            continue
        
        predefine_cpu = sch_task_lists[ids[0]]['resources.cpu']
        predefine_gpu = sch_task_lists[ids[0]]['resources.gpu']
        
        for task_id in ids:
            msg_size = sch_task_lists[task_id]['message_sizes.inputs']
            runtime =  Task_time_predictor.get_runtime(predefine_cpu, predefine_gpu, msg_size, name)
            
            cpu_area = predefine_cpu * runtime
            gpu_area = predefine_gpu * runtime
            task_details[task_cnt] = (name, task_id, predefine_cpu, predefine_gpu, "null", runtime, cpu_area, gpu_area, msg_size)
            task_cnt += 1
            
    for i, task in enumerate(np.sort(task_details, order=['gpu_area', 'cpu_area'])[::-1]):
        task_id = task['task_id']
        required_cpu = task['cpu']
        required_gpu = task['gpu']
        cpu_area = task['cpu_area']
        gpu_area = task['gpu_area']
        
        candidate_nodes = []
        for node_name, res in nodes.items():
            # 验证节点资源是否满足需求
            if (res['cpu'] >= required_cpu and 
                res['gpu'] >= required_gpu):
                candidate_nodes.append(node_name)
                
        if not candidate_nodes:
            print(f"任务 {task_id} 无法分配，资源不足")
            continue
        
        # 选择最佳节点（最小化最大资源利用率）
        best_node = None
        min_utilization = float('inf')
        
        for node in candidate_nodes:
            # 计算节点现有负载
            current_cpu_load = node_load[node]['cpu']
            current_gpu_load = node_load[node]['gpu']
            
            # 计算加入后的资源利用率
            cpu_util = (current_cpu_load + cpu_area) / nodes[node]['cpu']
            gpu_util = (current_gpu_load + gpu_area) / nodes[node]['gpu'] if nodes[node]['gpu']>0 else 1 # gpu_area = 0 in this situation
            max_util = max(cpu_util, gpu_util)
            
            # 选择利用率最低的节点
            if max_util < min_utilization:
                min_utilization = max_util
                best_node = node
        
        # 更新分配状态
        node_load[best_node]['cpu'] += cpu_area
        node_load[best_node]['gpu'] += gpu_area
        
        # 记录分配到节点
        # task_details[i]['node'] = best_node
        mask = task_details['task_id'] == task_id
        task_details['node'][mask] = best_node
    
    return task_details