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

# def balance_individual_tasks(ind: individual, 
#                            node_resources: dict, 
#                            time_predictor, 
#                            sch_task_list: dict, 
#                            max_iter: int = 20,
#                            verbose: bool = False) -> individual:
#     """
#     对individual的任务分配进行动态负载均衡优化
    
#     参数:
#         ind (individual): 需要优化的个体
#         node_resources (dict): 节点资源定义
#         time_predictor: 任务时间预测器
#         sch_task_list (dict): 任务元数据
#         max_iter (int): 最大优化迭代次数
#         verbose (bool): 是否输出调试信息

#     返回:
#         individual: 优化后的新个体（深拷贝）
#     """
#     # 创建副本避免修改原始个体
#     balanced_ind = ind.copy()
    
#     # 定义节点负载数据结构
#     NodeLoad = namedtuple('NodeLoad', ['cpu', 'gpu', 'tasks'])
#     node_states = defaultdict(lambda: NodeLoad(0, 0, []))
    
#     # 初始化节点状态
#     for task in balanced_ind.task_array:
#         node = task['node']
#         runtime = time_predictor.get_runtime(
#             task['cpu'], task['gpu'], 
#             sch_task_list[task['task_id']]['message_sizes.inputs'],
#             task['name']
#         )
#         cpu_load = task['cpu'] * runtime
#         gpu_load = task['gpu'] * runtime
        
#         node_states[node] = NodeLoad(
#             node_states[node].cpu + cpu_load,
#             node_states[node].gpu + gpu_load,
#             node_states[node].tasks + [task]
#         )
    
#     # 负载均衡主循环
#     for _ in range(max_iter):
#         # 计算当前各节点负载指标
#         load_metrics = {}
#         for node, load in node_states.items():
#             res = node_resources[node]
#             cpu_util = load.cpu / (res['cpu'] * 1e9)  # 假设CPU资源单位为GHz·s
#             gpu_util = load.gpu / (res['gpu'] * 1e9) if res['gpu'] > 0 else 0
#             load_metrics[node] = max(cpu_util, gpu_util)
        
#         # 寻找热点节点和冷节点
#         sorted_nodes = sorted(load_metrics.items(), key=lambda x: x[1], reverse=True)
#         hottest_node = sorted_nodes[0][0]
#         coldest_node = sorted_nodes[-1][0]
        
#         # 终止条件：最大负载差异小于5%
#         if (load_metrics[hottest_node] - load_metrics[coldest_node]) < 0.05:
#             break
            
#         # 从热点节点选择迁移候选任务（按资源消耗降序）
#         candidate_tasks = sorted(
#             node_states[hottest_node].tasks,
#             key=lambda t: (t['cpu']*t['total_runtime'] + t['gpu']*t['total_runtime']),
#             reverse=True
#         )
        
#         # 尝试迁移任务
#         migrated = False
#         for task in candidate_tasks:
#             # 寻找最佳目标节点
#             target_node = None
#             best_balance = float('inf')
            
#             for node in node_resources.keys():
#                 if node == hottest_node:
#                     continue
                
#                 # 检查资源约束
#                 if (task['cpu'] <= node_resources[node]['cpu'] and 
#                     task['gpu'] <= node_resources[node]['gpu']):
                    
#                     # 预测迁移后的负载平衡度
#                     new_hot_load = load_metrics[hottest_node] - (
#                         task['cpu']*task['total_runtime']/(node_resources[hottest_node]['cpu']*1e9) +
#                         task['gpu']*task['total_runtime']/(node_resources[hottest_node]['gpu']*1e9)
#                     )
#                     new_cold_load = load_metrics[node] + (
#                         task['cpu']*task['total_runtime']/(node_resources[node]['cpu']*1e9) +
#                         task['gpu']*task['total_runtime']/(node_resources[node]['gpu']*1e9)
#                     )
#                     balance = abs(new_hot_load - new_cold_load)
                    
#                     if balance < best_balance:
#                         best_balance = balance
#                         target_node = node
                        
#             if target_node and best_balance < (load_metrics[hottest_node] - load_metrics[coldest_node]):
#                 # 执行迁移
#                 task_idx = balanced_ind.get_task_index(task['task_id'])
#                 balanced_ind.task_array[task_idx]['node'] = target_node
                
#                 # 更新节点状态
#                 old_node = hottest_node
#                 new_node = target_node
                
#                 # 移除原节点负载
#                 runtime = task['total_runtime']
#                 node_states[old_node] = NodeLoad(
#                     node_states[old_node].cpu - task['cpu']*runtime,
#                     node_states[old_node].gpu - task['gpu']*runtime,
#                     [t for t in node_states[old_node].tasks if t['task_id'] != task['task_id']]
#                 )
                
#                 # 添加新节点负载
#                 node_states[new_node] = NodeLoad(
#                     node_states[new_node].cpu + task['cpu']*runtime,
#                     node_states[new_node].gpu + task['gpu']*runtime,
#                     node_states[new_node].tasks + [task]
#                 )
                
#                 migrated = True
#                 if verbose:
#                     print(f"Migrated task {task['task_id']} from {old_node} to {new_node}")
#                 break
                
#         if not migrated:
#             break  # 无法进一步优化
            
#     # 更新个体结构
#     balanced_ind.update_task_id_index()
#     balanced_ind.init_node_array()
    
#     return balanced_ind
