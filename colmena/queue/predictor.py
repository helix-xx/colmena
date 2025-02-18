from typing import Dict, List, Any, Optional, Union, Collection, Literal
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class TaskTimePredictor:
    methods: List[str]
    features: Dict[str, List[str]]
    
    _models: Dict[str, Pipeline] = {}
    _feature_scalers: Dict[str, StandardScaler] = {}
    
    # 缓存训练数据的统计信息
    _feature_stats: Dict[str, Dict[str, tuple]] = {}
    
    
    
    def __init__(self, methods, features, model_for_method:Dict[str,str] = {"run_calculator": "Polynomial", "run_sampling":"random_forest","train":"random_forest","evaluate":"random_forest"}):
        """
        初始化任务时间预测器
        
        Args:
            methods: 方法列表
            features: 特征列表
            model_for_method: 每个method的模型类型，可选 "random_forest" 或 "polynomial"
        """
        self.features = features
        self.methods = methods
            
        self.time_running_db = {}
        self.dtype = np.dtype([
            ('cpu', np.int32),
            ('gpu', np.int32),
            # ('node', 'U32'),
            ('message_sizes', np.float64),
            ('time_running', np.float64)
        ])
        
        # 为每个方法初始化模型
        for method, model_type in model_for_method.items():
            self._init_model(method, model_type)
        
        # 为每个方法初始化数据库
        for method in methods:
            self.time_running_db[method] = np.array([], dtype=self.dtype)
            
        # 用于快速查找的索引
        self.method_indices = {method: {} for method in methods}

    def _create_index(self, method: str):
        """创建索引以加速查询"""
        data = self.time_running_db[method]
        if len(data) == 0:
            return
            
        # 创建资源配置到索引的映射
        self.method_indices[method] = {
            # (row['cpu'], row['gpu'], row['node']): idx 
            (row['cpu'], row['gpu']): idx
            for idx, row in enumerate(data)
        }
            
            
    def fill_time_running_database(self, node_resources: Dict[str, Dict], 
                                 historical_data: Dict[str, List[Dict]]):
        """填充运行时间数据库"""
        for method in self.methods:
            # 收集所有唯一的message_sizes值
            message_sizes = set()
            for task in historical_data[method]:
                msg_size = task.get('message_sizes.inputs')
                if msg_size is not None:
                    message_sizes.add(float(msg_size))
            # 获得节点中的最大资源
            max_cpu = max(node["cpu"] for node in node_resources.values())
            max_gpu = max(node["gpu"] for node in node_resources.values())
            
            # 生成所有可能的配置组合
            records = []
            # for node, resources in node_resources.items():
            cpu_range = range(1, max_cpu + 1)
            gpu_range = range(0, max_gpu + 1)
            
            for msg_size in message_sizes:
                for cpu in cpu_range:
                    for gpu in gpu_range:
                        records.append((
                            cpu, gpu, msg_size, np.nan
                        ))
            
            # 使用numpy结构化数组存储数据
            self.time_running_db[method] = np.array(records, dtype=self.dtype)
            self._create_index(method)

    def fill_features_from_new_task(self, node_resources: Dict[str, Dict], sch_task_list: Dict):
        """从新任务中填充特征信息并合并到现有数据库"""
        
        for method in self.methods:
            # 收集所有新的 message_sizes 值
            message_sizes = set()
            for task_id, task in sch_task_list.items():
                if task['method'] == method:
                    msg_size = task.get('message_sizes.inputs')
                    if msg_size is not None:
                        message_sizes.add(float(msg_size))
            
            # 获得节点中的最大资源
            max_cpu = max(node["cpu"] for node in node_resources.values())
            max_gpu = max(node["gpu"] for node in node_resources.values())
            
            # 生成所有可能的新配置组合
            new_records = []
            cpu_range = range(1, max_cpu + 1)
            gpu_range = range(0, max_gpu + 1)
            
            for msg_size in message_sizes:
                for cpu in cpu_range:
                    for gpu in gpu_range:
                        new_records.append((
                            cpu, gpu, msg_size, np.nan
                        ))
            
            # 将新记录转换为 numpy 结构化数组
            new_array = np.array(new_records, dtype=self.dtype)
            
            # 合并到现有数据库中
            if method in self.time_running_db:
                existing_array = self.time_running_db[method]
                # 使用 np.concatenate 合并
                combined_array = np.concatenate((existing_array, new_array))
                # 去重，确保唯一性
                self.time_running_db[method] = np.unique(combined_array, axis=0)
            else:
                # 如果当前方法尚无记录，直接赋值
                self.time_running_db[method] = new_array
            
            # 更新索引
            self._create_index(method)

                                
    def add_runtime_records(self, historical_data: Dict[str, List[Dict]]):
        """批量添加历史运行记录"""
        for method, tasks in historical_data.items():
            if not tasks:
                continue
                
            data = self.time_running_db[method]
            indices = self.method_indices[method]
            
            for task in tasks:
                cpu = task['resources.cpu']
                gpu = task['resources.gpu']
                # node = task['resources']['node']
                msg_size = task['message_sizes.inputs']
                
                # 使用索引快速定位记录
                key = (cpu, gpu)
                if key in indices:
                    mask = (data['message_sizes'] == msg_size) & \
                           (data['cpu'] == cpu) & \
                           (data['gpu'] == gpu)
                        #    (data['node'] == node)
                    if np.any(mask):
                        data['time_running'][mask] = task['time_running']
                        
    def fill_runtime_records_with_predictor(self, method=None):
        """通过预测模型补充数据库缺失值"""
        for method in self.methods:
            if method not in self._models:
                logger.info(f"predictor for method{method} not trained")
                continue
                
            data = self.time_running_db[method]
            
            # 找到所有缺失值的位置
            missing_mask = np.isnan(data['time_running'])
            missing_indices = np.where(missing_mask)[0]
            
            if len(missing_indices) == 0:
                logger.info(f"no NAN for method{method}")
                continue
                
            # 准备特征数据进行批量预测
            features = np.zeros((len(missing_indices), 3))
            features[:, 0] = data['message_sizes'][missing_indices]
            features[:, 1] = data['cpu'][missing_indices]
            features[:, 2] = data['gpu'][missing_indices]
            
            try:
                # 批量预测
                predictions = np.expm1(self._models[method].predict(features))
                # 填充预测值
                data['time_running'][missing_indices] = predictions
            except Exception as e:
                logger.error(f"Prediction failed for method {method}: {str(e)}")
                continue


        
    @lru_cache(maxsize=10000) 
    def get_runtime(self, cpu, gpu, msg_size, method) -> Optional[float]:
        """获取运行时间"""
        if method not in self.methods:
            return None
            
        data = self.time_running_db[method]
        indices = self.method_indices[method]
        
        # 使用索引快速查找匹配的资源配置
        key = (cpu, gpu)
        if key not in indices:
            return None
            
        # 找到最接近的message_sizes值
        mask = (data['cpu'] == cpu) & \
                (data['gpu'] == gpu) 
        matches = data[mask]
        
        if len(matches) == 0:
            raise KeyError(f"get_runtime failed for features{(cpu, gpu, msg_size, method)}")
            
        # 找到最接近的message_sizes值对应的运行时间
        idx = np.abs(matches['message_sizes'] - msg_size).argmin()
        return float(matches[idx]['time_running'])

            
    def _init_model(self, method: str, model_type: Literal["random_forest", "polynomial"]):
        from sklearnex import patch_sklearn, unpatch_sklearn
        patch_sklearn()
        """为每个方法初始化模型pipeline"""
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=3)),
                ('linear_model', LinearRegression())
            ])
            
        self._models[method] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
    def train(self, train_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """统一的训练入口，返回每个方法的训练得分"""
        scores = {}
        for method, data in train_data.items():
            if len(data) < 5:  # 数据太少，跳过训练
                continue
                
            # 转换为numpy数组以提高性能
            X, y = self._prepare_training_data(data)
            if X is None or y is None:
                continue
                
            # 训练并记录得分
            try:
                self._models[method].fit(X, y)
                scores[method] = self._models[method].score(X, y)
            except Exception as e:
                logger.error(f"Training failed for method {method}: {str(e)}")
                continue
                
        return scores
    
    def _prepare_training_data(self, data: List[Dict]) -> tuple:
        """准备训练数据，返回(X, y)"""
        try:
            df = pd.DataFrame(data)
            df = df.dropna()
            
            # 提取特征和目标值
            X = df.drop(columns=['time_running', 'method', 'task_id'])
            y = np.log1p(df['time_running'])
            
            # 缓存特征统计信息供后续使用
            self._cache_feature_stats(df)
            
            return X.to_numpy(), y.to_numpy()
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return None, None
            
    def _cache_feature_stats(self, df: pd.DataFrame):
        """缓存特征统计信息"""
        for col in df.columns:
            if col not in ['time_running', 'method', 'task_id']:
                self._feature_stats[col] = (
                    float(df[col].mean()),
                    float(df[col].std())
                )
                
    @lru_cache(maxsize=1024)
    def _predict_single(self, method: str, feature_tuple: tuple) -> float:
        """预测单个任务的运行时间，使用tuple作为缓存key"""
        try:
            X = np.array(feature_tuple).reshape(1, -1)
            prediction = self._models[method].predict(X)[0]
            return np.expm1(prediction)
        except Exception as e:
            logger.error(f"Prediction failed for method {method}: {str(e)}")
            return None

    def estimate_time(self, task: Dict) -> Optional[float]:
        """预测单个任务的运行时间"""
        method = task['method']
        if method not in self._models:
            return None
            
        # 将特征转换为可哈希的tuple用于缓存
        features = self._extract_features(task)
        feature_tuple = tuple(features.values())
        
        return self._predict_single(method, feature_tuple)

    
    # def estimate_ga_population(self, population: List, sch_task_list: Dict, 
    #                         all_node: bool = False) -> None:
    #     """批量预测种群中所有任务的运行时间"""
        
    #     all_tasks = np.concatenate([ind.task_array for ind in population])
        
    #     method_to_tasks = {
    #         method: all_tasks[all_tasks['name'] == method] 
    #         for method in self.methods if method in self._models
    #     }
        
    #     for method, tasks in method_to_tasks.items():
    #         if len(tasks) == 0:
    #             continue
            
    #         try:
    #             num_tasks = len(tasks)
    #             # 一次性创建特征矩阵
    #             X = np.zeros((num_tasks, 3), dtype=np.float32)
                
    #             # 批量填充特征矩阵
    #             task_ids = tasks['task_id']
    #             X[:, 0] = [sch_task_list[tid].get('message_sizes.inputs', 1) for tid in task_ids]  # message sizes
    #             X[:, 1] = tasks['cpu']  # CPU
    #             X[:, 2] = tasks['gpu']  # GPU
                
    #             # 批量预测
    #             predictions = np.expm1(self._models[method].predict(X))
                
    #             # 创建任务ID到预测结果的映射
    #             prediction_map = dict(zip(task_ids, predictions))
                
    #             # 使用向量化操作更新种群
    #             for ind in population:
    #                 method_mask = ind.task_array['name'] == method
    #                 if np.any(method_mask):
    #                     task_ids = ind.task_array[method_mask]['task_id']
    #                     # 使用向量化操作更新运行时间
    #                     update_indices = [ind._task_id_index[tid] for tid in task_ids]
    #                     ind.task_array['total_runtime'][update_indices] = \
    #                         [prediction_map[tid] for tid in task_ids]
                        
    #         except Exception as e:
    #             logger.error(f"Method {method} batch预测失败: {str(e)}")
    #             logger.exception(e)
    
    def estimate_ga_population(self, population: List, sch_task_list: Dict, 
                         all_node: bool = False) -> None:
        """批量预测种群中所有任务的运行时间，优先使用数据库"""
        # 按方法分组处理任务
        method_tasks = defaultdict(list)
        for ind in population:
            for task in ind.task_array:
                method_tasks[task['name']].append((task, ind))
        
        # 批量处理每个方法的任务
        for method, tasks in method_tasks.items():
            if method not in self.methods:
                continue
                
            # 收集需要模型预测的任务
            model_prediction_tasks = []
            
            for task, ind in tasks:
                cpu = task['cpu']
                gpu = task['gpu']
                # 'node': task['node']
                
                msg_size = sch_task_list[task['task_id']].get(
                    'message_sizes.inputs', 0)
                    
                runtime = self.get_runtime(cpu, gpu, msg_size, method)
                
                if runtime is not None:
                    idx = ind._task_id_index[task['task_id']]
                    ind.task_array[idx]['total_runtime'] = runtime
                
    def _extract_features(self, task: Dict) -> Dict:
        """提取任务特征"""
        features = {}
        for feature in self.features['default']:
            if feature not in ['time_running', 'method', 'task_id']:
                value = self._get_nested_value(task, feature)
                features[feature] = value
        return features
    
    @staticmethod
    def _get_nested_value(dict_obj: Dict, key_path: str) -> Any:
        """获取嵌套字典中的值"""
        current = dict_obj
        for key in key_path.split('.'):
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current
    
    # def _extract_features_from_sch_task(self, sch_task: Dict, 
    #                                   resources: Dict) -> Dict:
    #     """从调度任务中提取特征"""
    #     features = sch_task.copy()
    #     # 更新资源相关的特征
    #     features['resources.cpu'] = resources['cpu']
    #     features['resources.gpu'] = resources['gpu']
    #     return features

    
    # def save_models(self, path: str):
    #     """保存训练好的模型"""
    #     import joblib
    #     save_dict = {
    #         'models': self._models,
    #         'feature_stats': self._feature_stats,
    #         'model_type': self.model_type
    #     }
    #     joblib.dump(save_dict, path)
    
    # @classmethod
    # def load_models(cls, path: str, methods: List[str], 
    #                features: Dict[str, List[str]]):
    #     """加载保存的模型"""
    #     import joblib
    #     saved_dict = joblib.load(path)
        
    #     predictor = cls(
    #         methods=methods,
    #         features=features,
    #         model_type=saved_dict['model_type']
    #     )
    #     predictor._models = saved_dict['models']
    #     predictor._feature_stats = saved_dict['feature_stats']
    #     return predictor
