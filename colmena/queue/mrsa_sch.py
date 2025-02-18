import os
import subprocess
import json
import logging

import numpy as np 
import pandas as pd 
from pandas import DataFrame 
from sklearn.linear_model import LinearRegression

from .monitor import Sch_data

logger = logging.getLogger(__name__)

def run_mrsa_scheduler(sch_data:Sch_data, model_type="powSum"):
    """使用MRSA替代GA进行调度"""
    # 准备输入文件
    folder_name = prepare_mrsa_input(sch_data, model_type)
    folder_name = "fitune_surrogate"
    
    # 获取资源配置
    first_node = list(sch_data.available_resources.keys())[0]
    total_cpu = sch_data.available_resources[first_node]['cpu']
    total_gpu = sch_data.available_resources[first_node]['gpu']
    
    # 调用MRSA调度器
    mrsa_path = sch_data.usr_path + "/project/colmena/multisite_/mrsa"
    cmd = f"python3 {mrsa_path}/alphaMaster.py {folder_name} 2 {total_cpu} {total_gpu} {model_type} {folder_name}"
    logger.info(f'Running MRSA scheduler with command: {cmd}')
    print(cmd)
    # os.system(cmd)
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        
        # 打印输出
        logger.info(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f'Command "{cmd}" failed with return code {e.returncode}')
        logger.error(f'Standard output: {e.stdout}')
        logger.error(f'Standard error: {e.stderr}')
    
    # 读取调度结果
    task_allocation = parse_mrsa_output(folder_name, sch_data, sch_data.available_resources)
    logger.info(f'MRSA scheduler finished with {len(task_allocation)} tasks allocated')
    
    return task_allocation

def convert_to_mrsa_models(task_list, models, model_type, output_folder="fitune_surrogate", dimensions=2):
    """将机器学习模型转换为MRSA支持的四种性能模型
    
    Args:
        task_list: 任务列表
        models: 训练好的ML模型（多项式或随机森林）
        model_type: 'amdSum'|'amdMax'|'powSum'|'powMax'
        output_folder: 输出目录
        dimensions: 资源维度数
    """
    def estimate_model_params(task, ml_model, model_type):
        """估计单个任务的模型参数"""
        # 采样点设置
        cpu_points = np.linspace(1, 32, 10)
        gpu_points = np.linspace(1, 4, 5)   
        
        base_cpu = 1
        base_gpu = 0
        if task.get('resources.gpu', 0) >= 1:
            base_gpu = 1
        # 1. 估计串行部分 s0
        X_serial = DataFrame([{
            'message_sizes.inputs': task.get('message_sizes.inputs', 1),
            'resources.cpu': base_cpu,
            'resources.gpu': base_gpu
        }])
        serial_time = max(1, np.expm1(ml_model.predict(X_serial)[0]))
        
        # 2. 分别采样CPU和GPU的影响
        times_cpu = []  # 仅CPU配置
        times_gpu = []  # CPU+GPU配置
        configs_cpu = []
        configs_gpu = []
        
        logger.info(f'Estimating model parameters for task {task["task_id"]}, serial time: {serial_time}')
        # CPU采样
        for cpu in cpu_points:
            X = DataFrame([{
                'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                'resources.cpu': cpu,
                'resources.gpu': base_gpu
            }])
            time = max(1, np.expm1(ml_model.predict(X)[0]))
            logger.info(f'CPU sample: CPU={cpu}, time={time}, features={X.to_dict(orient="records")}')
            times_cpu.append(time)
            configs_cpu.append([cpu])
        
        # GPU采样（固定最优CPU）如果原任务没有GPU，则不进行GPU采样
        if base_gpu != 0:
            optimal_cpu = cpu_points[np.argmin(times_cpu)]
            for gpu in gpu_points[1:]:  # 跳过gpu=0
                X = DataFrame([{
                    'message_sizes.inputs': task.get('message_sizes.inputs', 1),
                    'resources.cpu': optimal_cpu,
                    'resources.gpu': gpu
                }])
                time = max(1, np.expm1(ml_model.predict(X)[0]))
                logger.info(f'GPU sample: CPU={cpu}, time={time}, features={X.to_dict(orient="records")}')
                times_gpu.append(time)
                configs_gpu.append([optimal_cpu, gpu])
            
        # 3. 根据不同模型类型估计参数
        params = {}
        # 估计串行比例
        s0 = min(times_cpu)  # 基础串行时间
        
        # CPU部分
        X_cpu = np.array([1/c for c in cpu_points]).reshape(-1, 1)
        y_cpu = np.array(times_cpu) - s0
        reg_cpu = LinearRegression()
        reg_cpu.fit(X_cpu, y_cpu)
        s1 = max(0.1, reg_cpu.coef_[0])  # CPU并行部分权重
        
        # GPU部分
        if len(times_gpu) > 0:
            X_gpu = np.array([1/g for g in gpu_points[1:]]).reshape(-1, 1)
            # y_gpu = np.array(times_gpu) - s0 - s1/optimal_cpu
            y_gpu = np.array(times_gpu) - s0
            reg_gpu = LinearRegression()
            reg_gpu.fit(X_gpu, y_gpu)
            s2 = max(0, reg_gpu.coef_[0])  # GPU并行部分权重
        else:
            s2 = 0
            
        params = {
            's0': s0,
            's1': s1,
            's2': s2,
            'a1': 1.0,  # Amdahl模型固定为1
            'a2': 1.0
        }
        if model_type.startswith('amd'):  # Amdahl模型
            return params
            
        else:  # Power模型
            s0_amdahl = max(0, reg_cpu.intercept_)
            serial_time = min(s0-2, s0_amdahl) # 避免0或-inf
            # CPU部分
            X_cpu = np.log([c for c in cpu_points]).reshape(-1, 1)
            y_cpu = np.log(np.array(times_cpu) - serial_time)
            reg_cpu = LinearRegression()
            reg_cpu.fit(X_cpu, y_cpu)
            s1 = np.exp(reg_cpu.intercept_)
            a1 = -reg_cpu.coef_[0]
            
            # GPU部分
            if len(times_gpu) > 0:
                X_gpu = np.log(gpu_points[1:]).reshape(-1, 1)
                logger.info(f'GPU debug!! times_gpu: {times_gpu}, serial_time: {serial_time}, s1: {s1}, a1: {a1}')
                # y_gpu = np.log(np.array(times_gpu) - serial_time - s1/(optimal_cpu**a1))
                y_gpu = np.log(np.array(times_gpu) - serial_time)
                reg_gpu = LinearRegression()
                reg_gpu.fit(X_gpu, y_gpu)
                s2 = np.exp(reg_gpu.intercept_)
                a2 = -reg_gpu.coef_[0]
            else:
                s2 = 0
                a2 = 1.0
                
            params = {
                's0': serial_time,
                's1': max(0.1, s1),
                's2': max(0, s2),
                'a1': max(0.1, min(a1, 1.0)),
                'a2': max(0.1, min(a2, 1.0))
            }
            
        return params
    
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
            method = task['method']
            ml_model = models[method]
            
            # 估计参数
            params = estimate_model_params(task, ml_model, model_type)
            
            # 验证转换精度
            mean_err, max_err = validate_conversion(task, ml_model, params, model_type)
            conversion_stats['mean_error'].append(mean_err)
            conversion_stats['max_error'].append(max_err)
            conversion_stats['task_ids'].append(task['task_id'])
            
            # 写入MRSA格式
            line = f"{task['task_id']} {params['s0']} {params['s1']} {params['s2']} {params['a1']} {params['a2']}\n"
            f.write(line)
            
    return conversion_stats

def prepare_mrsa_input(sch_data, model_type="powSum", output_folder="fitune_surrogate"):
    """准备MRSA调度器输入"""
    # 获取当前需要调度的任务
    tasks = []
    for task_id, task in sch_data.sch_task_list.items():
        tasks.append(task)
    
    # if sch_data.Task_time_predictor.model_type == "random_forest":
    #     models = sch_data.Task_time_predictor.random_forest_models
    # elif sch_data.Task_time_predictor.model_type == "polynomial":
    #     models = sch_data.Task_time_predictor.polynomial_models
    
    models = sch_data.Task_time_predictor._models
    # 转换为power-max格式
    convert_to_mrsa_models(
        task_list=tasks,
        models=models,
        model_type=model_type,
        output_folder=output_folder
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
        
def parse_mrsa_output(output_folder, sch_data, node_resources):
    """解析MRSA的输出结果并转换为task_allocation格式
    
    Args:
        output_folder: MRSA输出文件夹名称
        sch_data: 调度器数据对象
        node_resources: 节点资源信息
    
    Returns:
        task_allocation: 列表形式的任务分配方案
    """
    task_allocation = []

    which_node = generate_node_cycle(node_resources.keys())
    
    # 读取MRSA的输出文件
    usr_path = os.path.expanduser('~')
    with open(f"{usr_path}/project/colmena/multisite_/mrsa/files/allocation/{output_folder}/sample0seq.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            task_id = parts[0]
            cpu_count = int(parts[1])
            gpu_count = int(parts[2])
            node1_res = node_resources['node1']
            if cpu_count > node1_res['cpu']:
                cpu_count = node1_res['cpu']
            # 获取任务的method
            task = sch_data.sch_task_list.get(task_id)
            if not task:
                continue
                
            # 简单的循环分配节点
            node = next(which_node)
            
            # 创建task_allocation格式的任务描述
            new_task = {
                "name": task['method'],
                "task_id": task_id,
                "resources": {
                    "cpu": cpu_count,
                    "gpu": gpu_count,
                    "node": node
                }
            }
            task_allocation.append(new_task)
    
    return task_allocation