import dis
import os
import subprocess
import json
import logging

import numpy as np 
import pandas as pd 
from pandas import DataFrame 
from sklearn.linear_model import LinearRegression
from collections import defaultdict

from .monitor import Sch_data
from .evo_sch import individual


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
        

def run_mrsa_scheduler(sch_data:Sch_data, model_type="powSum", tasks=None):
    """使用MRSA替代GA进行调度"""
    # 负载均衡以支持多节点
    node_tasks = distribute_tasks(tasks, sch_data.available_resources, sch_data.sch_task_list, sch_data.Task_time_predictor)
    
    for node_name, resources in sch_data.available_resources.items():
        node_task = node_tasks[node_tasks['node'] == node_name]
        if len(node_task) == 0:
            logger.info(f"节点 {node_name} 没有分配任务，跳过")
            continue
            
        folder_name = f"mrsa_{node_name}"
        os.makedirs(folder_name, exist_ok=True)
        
        try:
            node_res = sch_data.available_resources[node_name]
            total_cpu = node_res['cpu']
            total_gpu = node_res['gpu']
            
            # 动态确定维度数量
            dimensions = 1 if total_gpu == 0 else 2
            
            # 准备输入文件时传入维度信息
            prepare_mrsa_input(
                sch_data=sch_data, 
                tasks=node_task, 
                model_type=model_type, 
                output_folder=folder_name,
                dimensions=dimensions
            )
            
            mrsa_path = sch_data.usr_path + "/project/colmena/multisite_/mrsa"
            cmd = f"python3 {mrsa_path}/alphaMaster.py {folder_name} {dimensions} {total_cpu} "
            
            # 只有在有GPU资源时才添加GPU参数
            if dimensions == 2:
                cmd += f"{total_gpu} "
                
            cmd += f"{model_type} {folder_name}"
            
            logger.info(f'Running MRSA scheduler with command: {cmd}')
            print(cmd)
            
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
            print(result.stdout)
            print(result.stderr)
            
            # 解析输出时传入维度信息
            parse_mrsa_output(
                output_folder=folder_name, 
                sch_data=sch_data, 
                distributed_tasks=node_task, 
                dimensions=dimensions
            )
            
        except Exception as e:
            logger.error(f"节点 {node_name} 调度失败: {str(e)}")
            logger.exception(e)
            continue

    logger.info(f'MRSA scheduler finished, scheduled tasks {node_tasks}')
    
    mrsa_ind = individual(tasks_nums=len(node_tasks), total_resources=sch_data.available_resources)
    mrsa_ind.task_array = np.array(
            [(t['name'], t['task_id'], 
              t['cpu'], t['gpu'], 
              t['node'], t['total_runtime'],
              0, 0)
             for t in node_tasks],
            dtype=mrsa_ind.dtype
        )
    mrsa_ind.update_task_id_index()
    sch_data.best_ind = mrsa_ind  # type: ignore[attr-defined]
    return mrsa_ind.task_array

def convert_to_mrsa_models(task_list, models, model_type, output_folder="fitune_surrogate", dimensions=2):
    """将机器学习模型转换为MRSA支持的四种性能模型
    
    Args:
        task_list: 任务列表
        models: 训练好的ML模型（多项式或随机森林）
        model_type: 'amdSum'|'amdMax'|'powSum'|'powMax'
        output_folder: 输出目录
        dimensions: 资源维度数
    """
    
    # 主处理流程
    usr_path = os.path.expanduser('~')
    os.makedirs(f"{usr_path}/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}", exist_ok=True)
    output_file = f"{usr_path}/project/colmena/multisite_/mrsa/files/tasks_parameters/{output_folder}/sample0.txt"
    
    conversion_stats = {
        'mean_error': [],
        'max_error': [],
        'task_ids': []
    }
    
    with open(output_file, 'w') as f:
        for task in task_list:
            # 估计参数
            params = estimate_model_params(task, models, model_type)
            
            # 根据维度生成不同格式的输出行
            if dimensions == 1:
                # 1维资源(只有CPU)格式
                line = f"{task['task_id']} {params['s0']} {params['s1']} {params['a1']}\n"
            else:
                # 2维资源(CPU+GPU)格式
                line = f"{task['task_id']} {params['s0']} {params['s1']} {params['s2']} {params['a1']} {params['a2']}\n"
            
            f.write(line)
            
    return conversion_stats

def prepare_mrsa_input(sch_data, tasks, model_type="powSum", output_folder="fitune_surrogate", dimensions=2):
    """准备MRSA调度器输入"""
    
    models = sch_data.Task_time_predictor
    # 转换为power-max格式，传入维度参数
    convert_to_mrsa_models(
        task_list=tasks,
        models=models,
        model_type=model_type,
        output_folder=output_folder,
        dimensions=dimensions
    )
    
    # 创建空的依赖关系文件夹和文件(因为不考虑依赖关系)
    usr_path = os.path.expanduser('~')
    os.makedirs(f"{usr_path}/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}", exist_ok=True)
    with open(f"{usr_path}/project/colmena/multisite_/mrsa/files/precedence_constraints/{output_folder}/sample0.txt", 'w') as f:
        pass
    
    # 创建分配结果文件夹
    os.makedirs(f"{sch_data.usr_path}/project/colmena/multisite_/mrsa/files/allocation/{output_folder}", exist_ok=True)
    
    return output_folder

def generate_node_cycle(nodes):
    """生成循环节点选择器"""
    node_list = list(nodes)
    index = 0
    while True:
        yield node_list[index]
        index = (index + 1) % len(node_list)
        
def parse_mrsa_output(output_folder, sch_data, distributed_tasks, dimensions=2):
    """解析MRSA的输出结果并转换为task_allocation格式
    
    Args:
        output_folder: MRSA输出文件夹名称
        sch_data: 调度器数据对象
        distributed_tasks: 分布式任务信息
        dimensions: 资源维度数
    
    Returns:
        task_allocation: 列表形式的任务分配方案
    """
    # 读取MRSA的输出文件
    usr_path = os.path.expanduser('~')
    output_file = f"{usr_path}/project/colmena/multisite_/mrsa/files/allocation/{output_folder}/sample0seq.txt"
    
    if not os.path.exists(output_file):
        logger.error(f"MRSA输出文件不存在: {output_file}")
        return
    
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:  # 至少需要任务ID和CPU分配
                continue
                
            task_id = parts[0]
            cpu_count = int(parts[1])
            
            # 默认GPU为0，只有在二维情况下才从输出解析GPU数量
            gpu_count = 0
            if dimensions == 2 and len(parts) >= 3:
                gpu_count = int(parts[2])

            # 使用NumPy布尔索引精确匹配task_id并更新资源
            mask = distributed_tasks['task_id'] == task_id
            if np.any(mask):
                distributed_tasks['cpu'][mask] = cpu_count
                distributed_tasks['gpu'][mask] = gpu_count
                # 确保更新资源字典字段（如果存在）
                if 'resources.cpu' in distributed_tasks.dtype.names:
                    distributed_tasks['resources.cpu'][mask] = cpu_count
                if 'resources.gpu' in distributed_tasks.dtype.names:
                    distributed_tasks['resources.gpu'][mask] = gpu_count

# 验证函数
def validate_conversion(task, ml_model, params, model_type):
    """验证转换精度"""
    def predict_mrsa_time(cpu, gpu, params, model_type):
        if model_type == 'amdSum':
            return (params['s0'] + 
                    params['s1']/cpu + 
                    (params['s2']/gpu if gpu > 0 else 0))
        elif model_type == 'amdMax':
            return (params['s0'] + 
                    max(params['s1']/cpu,
                        params['s2']/gpu if gpu > 0 else 0))
        elif model_type == 'powSum':
            return (params['s0'] + 
                    params['s1']/(cpu**params['a1']) + 
                    (params['s2']/(gpu**params['a2']) if gpu > 0 else 0))
        else:  # powMax
            return (params['s0'] + 
                    max(params['s1']/(cpu**params['a1']),
                        params['s2']/(gpu**params['a2']) if gpu > 0 else 0))
    
    # 验证点
    cpu_test = np.linspace(1, 16, 8)
    gpu_test = np.linspace(0, 4, 5)
    
    errors = []
    for cpu in cpu_test:
        for gpu in gpu_test:
            # ML模型预测
            X = DataFrame([{
                'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                'resources.cpu': cpu,
                'resources.gpu': gpu
            }])
            ml_time = np.expm1(ml_model.predict(X)[0])
            
            # MRSA模型预测
            mrsa_time = predict_mrsa_time(cpu, gpu, params, model_type)
            
            error = abs(ml_time - mrsa_time) / ml_time
            errors.append(error)
    
    return np.mean(errors), np.max(errors)

def estimate_model_params(task, models, model_type):
    """估计单个任务的模型参数"""
    # 采样点设置
    cpu_points = np.linspace(1, 24, 24)
    gpu_points = np.linspace(1, 4, 4)
    
    base_cpu = 1
    base_gpu = 0
    if task['gpu'] >= 1:
        base_gpu = 1
    # 1. 估计串行部分 s0
    msg_size = task['feature_msg_size']
    method = task['name']

    serial_time = models.get_runtime(base_cpu, base_gpu, msg_size, method)
    
    # 2. 分别采样CPU和GPU的影响
    times_cpu = []  # 仅CPU配置
    times_gpu = []  # CPU+GPU配置
    configs_cpu = []
    configs_gpu = []
    
    logger.info(f'Estimating model parameters for task {task["task_id"]}, serial time: {serial_time}')
    # CPU采样
    for cpu in cpu_points:
        time = models.get_runtime(cpu, base_gpu, msg_size, method)
        logger.debug(f'CPU sample: CPU={cpu}, time={time}, task={task}')
        times_cpu.append(time)
        configs_cpu.append([cpu])
    
    # GPU采样（固定最优CPU）如果原任务没有GPU，则不进行GPU采样
    if base_gpu != 0:
        optimal_cpu = cpu_points[np.argmin(times_cpu)]
        for gpu in gpu_points[1:]:  # 跳过gpu=0
            time = models.get_runtime(optimal_cpu, gpu, msg_size, method)
            logger.debug(f'GPU sample: GPU={gpu}, time={time}, task={task}')
            times_gpu.append(time)
            configs_gpu.append([optimal_cpu, gpu])
        
    # 3. 根据不同模型类型估计参数
    # 估计串行比例
    s0 = min(times_cpu)  # 基础串行时间
    
    # CPU部分
    X_cpu = np.array([1/c for c in cpu_points]).reshape(-1, 1)
    y_cpu = np.array(times_cpu) - s0
    reg_cpu = LinearRegression()
    reg_cpu.fit(X_cpu, y_cpu)
    s1 = max(0.1, float(reg_cpu.coef_[0]))  # CPU并行部分权重
    
    # GPU部分 - 只有当任务需要GPU时才计算
    s2 = 0  # 默认GPU权重为0
    a2 = 1.0  # 默认GPU指数为1.0
    
    if task['gpu'] > 0 and len(times_gpu) > 0:
        X_gpu = np.array([1/g for g in gpu_points[1:]]).reshape(-1, 1)
        y_gpu = np.array(times_gpu) - s0
        reg_gpu = LinearRegression()
        reg_gpu.fit(X_gpu, y_gpu)
        s2 = max(0.0, float(reg_gpu.coef_[0]))  # GPU并行部分权重
        
    # 初始化参数
    params = {
        's0': s0,
        's1': s1,
        's2': s2,
        'a1': 1.0,  # Amdahl模型固定为1
        'a2': 1.0
    }
    
    # 如果是Power模型，需要额外计算指数
    if not model_type.startswith('amd'):  # Power模型
        s0_amdahl = max(0, float(reg_cpu.intercept_))
        serial_time = min(s0-2, s0_amdahl) # 避免0或-inf
        # CPU部分
        X_cpu = np.log([c for c in cpu_points]).reshape(-1, 1)
        y_cpu = np.log(np.array(times_cpu) - serial_time)
        reg_cpu = LinearRegression()
        reg_cpu.fit(X_cpu, y_cpu)
        s1 = np.exp(reg_cpu.intercept_)
        a1 = -reg_cpu.coef_[0]
        
        # GPU部分 - 只有当任务需要GPU时才计算
        if task['gpu'] > 0 and len(times_gpu) > 0:
            X_gpu = np.log(gpu_points[1:]).reshape(-1, 1)
            y_gpu = np.log(np.array(times_gpu) - serial_time)
            reg_gpu = LinearRegression()
            reg_gpu.fit(X_gpu, y_gpu)
            s2 = np.exp(reg_gpu.intercept_)
            a2 = -reg_gpu.coef_[0]
        
        params = {
            's0': serial_time,
            's1': max(0.1, s1),
            's2': max(0, s2),
            'a1': max(0.1, min(a1, 1.0)),
            'a2': max(0.1, min(a2, 1.0))
        }
        
    return params
