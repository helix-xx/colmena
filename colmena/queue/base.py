"""Base classes for queues and related functions"""
from functools import total_ordering
import queue
import re
import resource
import warnings
from abc import abstractmethod
from enum import Enum
import threading
from threading import Lock, Event
from typing import Callable, Optional, Tuple, Any, Collection, Union, Dict, Set
import logging
from attr import dataclass
from collections import defaultdict
import time

import proxystore.store

from colmena.models import Result, SerializationMethod, ResourceRequirements
from colmena.queue import evo_sch

logger = logging.getLogger(__name__)


class QueueRole(str, Enum):
    """Role a queue is used for"""

    ANY = 'any'
    SERVER = 'server'
    CLIENT = 'client'


class ColmenaQueues:
    """Base class for a queue used in Colmena.

    Follows the basic ``get`` and ``put`` semantics of most queues,
    with the addition of a "topic" used by Colmena to separate
    task requests or objects used for different purposes."""

    def __init__(self,
                 topics: Collection[str],
                 methods: Collection[str],
                 serialization_method: Union[str,
                                             SerializationMethod] = SerializationMethod.JSON,
                 keep_inputs: bool = True,
                 proxystore_name: Optional[Union[str, Dict[str, str]]] = None,
                 proxystore_threshold: Optional[Union[int,
                                                      Dict[str, int]]] = None,
                 available_task_capacity: Optional[int] = 16,
                 #  estimate_methods: Optional[Dict[str, callable]] = None,
                 enable_evo=False,
                 available_resources: dict = None):
        """
        Args:
            topics: Names of topics that are known for this queue
            serialization_method: Method used to serialize task inputs and results
            keep_inputs: Whether to return task inputs with the result object
            proxystore_name (str, dict): Name of a registered ProxyStore
                `Store` instance. This can be a single name such that the
                corresponding `Store` is used for all topics or a mapping of
                topics to registered `Store` names. If a mapping is provided
                but a topic is not in the mapping, ProxyStore will not be used.
            proxystore_threshold (int, dict): Threshold in bytes for using
                ProxyStore to transfer objects. Optionally can pass a dict
                mapping topics to threshold to use different threshold values
                for different topics. None values in the mapping will exclude
                ProxyStore use with that topic.

            available_task_capacity: the capacity of available task list, after
                the capacity is reached, the task this flag prevent the agent submit 
                task to the queue
        """

        # Store the list of topics and other simple options
        self.enable_evo = enable_evo
        self.topics = set(topics)
        self.methods = set(methods)
        self.topics.add('default')
        self.keep_inputs = keep_inputs
        self.serialization_method = serialization_method
        self.role = QueueRole.ANY

        # available task for evosch, YXX
        self._add_task_flag = Event()
        self._add_task_lock = Lock()
        self._add_task_flag.set()
        # TODO Extract historical data to estimate running time, and register estimate_methods. by YXX
        historical_data = evo_sch.historical_data(methods=methods, queue=self)
        self._available_task_capacity = available_task_capacity
        available_task = {}
        for method in self.methods:
            available_task[method] = []
        self._available_tasks = evo_sch.available_task(available_task)

        self.evosch_lock = threading.Lock()
        self.evosch: evo_sch.evosch2 = evo_sch.evosch2(
            resources=available_resources, at=self._available_tasks, hist_data=historical_data)
        self.best_allocation = None  # best allocation
        
        # TODO tmp test, acquire task while resources idle
        self.prepared_task = defaultdict(list)

        # TODO function, improvement weight
        # trace task submit seq

        # Result list temp for result object, can be quick search by task_id
        self.result_list = {}  # can be quick search by id

        # timer for trigger evo_sch
        timer = None

        def reset_timer():
            nonlocal timer
            # 重置计时器，取消之前的计时器并启动新的计时器
            if timer:
                timer.cancel()
            timeout = 3
            timer = threading.Timer(timeout, self.trigger_evo_sch)
            timer.start()
        self.timer_trigger = reset_timer

        # Create {topic: proxystore_name} mapping
        self.proxystore_name = {t: None for t in self.topics}
        if isinstance(proxystore_name, str):
            self.proxystore_name = {t: proxystore_name for t in self.topics}
        elif isinstance(proxystore_name, dict):
            self.proxystore_name.update(proxystore_name)
        elif proxystore_name is not None:
            raise ValueError(
                f'Unexpected type {type(proxystore_name)} for proxystore_name')

        # Create {topic: proxystore_threshold} mapping
        self.proxystore_threshold = {t: None for t in self.topics}
        if isinstance(proxystore_threshold, int):
            self.proxystore_threshold = {
                t: proxystore_threshold for t in self.topics}
        elif isinstance(proxystore_threshold, dict):
            self.proxystore_threshold.update(proxystore_threshold)
        elif proxystore_threshold is not None:
            raise ValueError(
                f'Unexpected type {type(proxystore_threshold)} for proxystore_threshold')

        # Verify that ProxyStore backends exist
        for ps_name in set(self.proxystore_name.values()):
            if ps_name is None:
                continue
            store = proxystore.store.get_store(ps_name)
            if store is None:
                raise ValueError(
                    f'A Store with name "{ps_name}" has not been registered. '
                    'This is likely because the store needs to be '
                    'initialized prior to initializing the Colmena queue.'
                )

        # Log the ProxyStore configuration
        for topic in self.topics:
            ps_name = self.proxystore_name[topic]
            ps_threshold = self.proxystore_threshold[topic]

            if ps_name is None or ps_threshold is None:
                logger.debug(f'Topic {topic} will not use ProxyStore')
            else:
                logger.debug(
                    f'Topic {topic} will use ProxyStore backend "{ps_name}" '
                    f'with a threshold of {ps_threshold} bytes'
                )

        # Create a collection that holds the task which have been sent out, and an event that is triggered
        #  when the last task being sent out hits zero
        self._active_lock = Lock()
        self._active_tasks: Set[str] = set()
        self._all_complete = Event()

    def __getstate__(self):
        state = self.__dict__.copy()
        # We do not send the lock or event over pickle
        state.pop('_active_lock')
        state.pop('_all_complete')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._active_lock = Lock()
        self._all_complete = Event()

    def _check_role(self, allowed_role: QueueRole, calling_function: str):
        """Check whether the queue is in an appropriate role for a requested function

        Emits a warning if the queue is in the wrong role

        Args:
            allowed_role: Role to check for
            calling_function: Name of the calling function
        """
        if self.role != QueueRole.ANY and self.role != allowed_role:
            warnings.warn(
                f'{calling_function} is intended for {allowed_role} not a {self.role}')

    @property
    def active_count(self) -> int:
        """Number of active tasks"""
        return len(self._active_tasks)

    def set_role(self, role: Union[QueueRole, str]):
        """Define the role of this queue.

        Controls whether users will be warned away from performing actions that are disallowed by
        a certain queue role, such as sending results from a client or issuing requests from a server"""
        role = QueueRole(role)
        self.role = role

    def get_result(self, topic: str = 'default', timeout: Optional[float] = None) -> Optional[Result]:
        self._check_role(QueueRole.CLIENT, 'get_result')

        message = self._get_result(timeout=timeout, topic=topic)
        logger.debug(f'Received value: {str(message)[:25]}')

        result_obj = Result.parse_raw(message)
        result_obj.time_deserialize_results = result_obj.deserialize()
        result_obj.mark_result_received()

        with self._active_lock:
            self._active_tasks.discard(result_obj.task_id)
            if len(self._active_tasks) == 0:
                self._all_complete.set()

        logger.info(
            f'Client received a {result_obj.method} result with topic {topic}, consume resource is {result_obj.inputs[1]}')

        if self.enable_evo:
            with self.evosch_lock:
                node = getattr(result_obj.resources, 'node')
                gpu_value = result_obj.inputs[1]['gpu']
                self.evosch.resources[node]['cpu'] += result_obj.inputs[1]['cpu']
                self.evosch.resources[node]['gpu'] += len(gpu_value)
                self.evosch.resources[node]['gpu_devices'].extend(gpu_value)
                self.evosch.resources[node]['gpu_devices'].sort()
                # setattr(result_obj.resources, 'gpu', len(gpu_value))
                for task in self.evosch.running_task_node[node]:
                    if task['task_id'] == result_obj.task_id:
                        self.evosch.running_task_node[node].remove(task)
                        break

                if result_obj.success:
                    self.evosch.hist_data.complete_task_seq.append(
                        {"method": result_obj.method, "topic": topic, "task_id": result_obj.task_id, "time": time.time(), "type": "complete"})
                    self.evosch.hist_data.get_features_from_result_object(
                        result_obj)
                else:
                    for item in self.evosch.hist_data.submit_task_seq:
                        if item['task_id'] == result_obj.task_id:
                            self.evosch.hist_data.submit_task_seq.remove(item)
                            break

                logger.info(
                    f'Client received a {result_obj.method} result with topic {topic}, restore resources: remaining resources on node {node} are {self.evosch.resources[node]}')

                if self.best_allocation:
                    self.trigger_submit_task(self.best_allocation)
                else:
                    self._add_task_flag.set()
                    if self.evosch.at.get_total_nums() > 0:
                        self.timer_trigger()

        return result_obj

    def send_inputs(self,
                    *input_args: Any,
                    method: str = None,
                    input_kwargs: Optional[Dict[str, Any]] = None,
                    keep_inputs: Optional[bool] = None,
                    resources: Optional[Union[ResourceRequirements, dict]] = None,
                    topic: str = 'default',
                    task_info: Optional[Dict[str, Any]] = None) -> str:
        """Send a task request

        Args:
            *input_args (Any): Positional arguments to a function
            method (str): Name of the method to run. Optional
            input_kwargs (dict): Any keyword arguments for the function being run
            keep_inputs (bool): Whether to override the
            topic (str): Topic for the queue, which sets the topic for the result
            resources: Suggestions for how many resources to use for the task
            task_info (dict): Any information used for task tracking
        Returns:
            Task ID
        """
        self._check_role(QueueRole.CLIENT, 'send_inputs')

        # TODO YXX modified here, add scheduler, trigger by send_inputs and get_result
        # if the task capacity is full, wait for the evo_sch to trigger, and finish some task
        self._add_task_flag.wait()
        # Make sure the queue topic exists
        assert topic in self.topics, f'Unknown topic: {topic}. Known are: {", ".join(self.topics)}'

        # Make fake kwargs, if needed
        if input_kwargs is None:
            input_kwargs = dict()

        # Determine whether to override the default "keep_inputs"
        _keep_inputs = self.keep_inputs
        if keep_inputs is not None:
            _keep_inputs = keep_inputs

        # Gather ProxyStore info if we are using it with this topic
        ps_name = self.proxystore_name[topic]
        ps_threshold = self.proxystore_threshold[topic]
        ps_kwargs = {}
        if ps_name is not None and ps_threshold is not None:
            store = proxystore.store.get_store(ps_name)
            # proxystore_kwargs contains all the information we would need to
            # reconnect to the ProxyStore backend on any worker
            ps_kwargs.update({
                'proxystore_name': ps_name,
                'proxystore_threshold': ps_threshold,
                'proxystore_config': store.config(),
            })

        # Create a new Result object
        # logger.info(f'input_args is {input_args}, input_kwargs is {input_kwargs}')
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
        
        
        if self.enable_evo:
            # TODO tmp test for acquire task while resources idle
            # check if we tmp save the task in dict
            if self.evosch.prepared_task[method] is None or len(self.prepared_task[method]) < 4:
                result_p = Result(
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
                result_p.time_serialize_inputs, proxies = result_p.serialize()
                self.evosch.prepared_task[method].append(result_p)
            
            # add to available task list YXX, under this lock agent cant submit task
            with self._add_task_lock:
                self._available_tasks.add_task_id(
                    task_name=method, task_id=result.task_id)
                self.evosch.hist_data.submit_task_seq.append(
                    {"method": method, "topic": topic, "task_id": result.task_id, "time": time.time(), "type": "submit"})
                logger.info(f'Client sent a {method} task with topic {topic}.')
                self.result_list[result.task_id] = result
                # detect the capacity
                # if self._available_tasks.get_total_nums() >= self._available_task_capacity:
                #     logger.info(f'Client reach the capacity.')
                #     self._add_task_flag.clear() # TODO for now we disable it to run the test
            self.timer_trigger()
        else:
            # Push the serialized value to the task server
            # move this after the task is added to the available task list and decide by scheduler
            self._send_request(result.json(exclude_none=True), topic)
            logger.info(
                f'Client sent a {method} task with topic {topic}. Created {len(proxies)} proxies for input values')

        # Store the task ID in the active list
        with self._active_lock:
            self._active_tasks.add(result.task_id)
            self._all_complete.clear()
        return result.task_id

    def trigger_submit_task(self, best_allocation: list):
        logger.info(
            f'Client trigger submit task, available task length is {len(best_allocation)}')

        # 创建一个字典来保存每个节点的任务队列
        node_task_queues = {}
        for task in best_allocation:
            node = task['resources']['node']
            if node not in node_task_queues:
                node_task_queues[node] = []
            node_task_queues[node].append(task)

        # 创建一个字典来记录每个节点是否被阻塞
        node_blocked = {node: False for node in node_task_queues.keys()}

        while True:
            all_nodes_blocked = True
            for node, tasks in node_task_queues.items():
                if node_blocked[node]:
                    continue  # 如果节点被阻塞，跳过这个节点

                if not tasks:
                    continue  # 如果没有任务，跳过这个节点

                task = tasks[0]
                cpu_value = task['resources']['cpu']
                gpu_value = task['resources']['gpu']

                if self.evosch.resources[node]['cpu'] < cpu_value or self.evosch.resources[node]['gpu'] < gpu_value:
                    logger.info(
                        f'Client trigger submit task, resource is not enough on node {node} for task {task["task_id"]}, marking this node as blocked')
                    node_blocked[node] = True  # 标记这个节点为阻塞状态
                else:
                    logger.info(
                        f"submit task {task['task_id']} to queue on node {node}, remaining resources are {self.evosch.resources[node]}, consume resources are {task['resources']}")
                    self.evosch.resources[node]['cpu'] -= cpu_value
                    self.evosch.resources[node]['gpu'] -= gpu_value

                    # 从原始 best_allocation 列表中移除该任务
                    best_allocation.remove(task)
                    tasks.pop(0)  # 从当前节点的任务队列中移除该任务

                    self.evosch.at.remove_task_id(
                        task_name=task['name'], task_id=task['task_id'])

                    predict_task = {
                        'name': task['name'],
                        'task_id': task['task_id'],
                        'start_time': time.time(),
                        'finish_time': None,
                        'total_runtime': task['total_runtime'],
                        'resources': task['resources'],
                        'node': node
                    }
                    predict_task['finish_time'] = predict_task['start_time'] + \
                        predict_task['total_runtime']
                    self.evosch.running_task_node[node].append(predict_task)

                    result = self.result_list.pop(task['task_id'])
                    result.inputs[1]['cpu'] = cpu_value
                    gpu_value, self.evosch.resources[node]['gpu_devices'] = self.evosch.resources[node][
                        'gpu_devices'][:gpu_value], self.evosch.resources[node]['gpu_devices'][gpu_value:]
                    result.inputs[1]['gpu'] = gpu_value
                    setattr(result.resources, 'cpu', cpu_value)
                    setattr(result.resources, 'gpu', gpu_value)
                    setattr(result.resources, 'node', node)
                    topic = result.topic
                    self._send_request(result.json(exclude_none=True), topic)
                    logger.info(
                        f'Resources: result.resources is: {result.resources}')

                    # 重置节点的阻塞状态
                    node_blocked[node] = False

            # 检查是否所有节点都被阻塞
            all_nodes_blocked = all(node_blocked[node] or not tasks for node, tasks in node_task_queues.items())
            if all_nodes_blocked:
                logger.info('All nodes are blocked or have no tasks to submit, stopping submission.')
                break

        # remain task are waiting for resources, every n seconds trigger submit
        if self.evosch.at.get_total_nums() > 0:
            self.timer_trigger()

    def trigger_evo_sch(self):
        """Conditions that trigger scheduling and submission of tasks

        Args:
            result (Result): _description_
            topic (str, optional): _description_. Defaults to 'default'.
        """
        if self.evosch_lock.acquire(blocking=False):
            # only one thread could trigger this
            logger.info("evosch_lock acquire")
            # run_flag = False
            try:
                if self._available_tasks.get_total_nums() == 0:
                    logger.info("no available task in the available task list")
                    return
                # no pending task on any node
                # check any node have no pending task, go into runga
                elif self.evosch.check_pending_task_on_node(self.best_allocation):
                    logger.info(
                        f'Client trigger evo_sch because no task pending on one or more node, available task is {self._available_tasks.task_ids}, pending task is {self.best_allocation}')
                else:
                    logger.info(
                        f'Client trigger evo_sch because task pending on all nodes, available task is {self._available_tasks.task_ids}, pending task is {self.best_allocation}')
                    return
                # TODO think correct logic
                # add condition here to trigger evo_sch
                # condition 1: send_inputs, then timetrigger timeout
                # condition 2: the task capacity is full? need re consider
                #
                # while workflow running, new task come and re sort the queue
                # a task completed, reset timetrigger
                #

                # check any node have no pending task, go into runga
                if self._add_task_flag.is_set() is False:
                    # get ind and submit task on the sch order and resources
                    # drop submitted task and record resources
                    logger.info(
                        f'Client trigger evo_sch because capacity is full, available task is {self._available_tasks.task_ids}')
                    logger.info(
                        f'result list length is {len(self.result_list)}')
                    # if self.evosch.resources['cpu']>=16:
                    #     pass
                    # else:
                    #     logger.info(f'Client trigger evo_sch because resource is not enough, wait for resource')
                    self.best_allocation = self.evosch.run_ga(
                        5)  # parameter may need modify
                    self.trigger_submit_task(self.best_allocation)

                if self._add_task_flag.is_set() is True:
                    logger.info(
                        f'Client trigger evo_sch because submit task time out, it may means that submit agent may block until submitted task is done')
                    logger.info(
                        f'result list length is {len(self.result_list)}')
                    self.best_allocation = self.evosch.run_ga(
                        100)  # parameter may need modify
                    self.trigger_submit_task(self.best_allocation)
            finally:
                self.evosch_lock.release()
                logger.info("evosch_lock release")
        else:
            logger.info("evosch_lock acquire fail, evosch is running")
            return

    def wait_until_done(self, timeout: Optional[float] = None):
        """Wait until all out-going tasks have completed

        Returns:
            Whether the event was set within the timeout
        """
        self._check_role(QueueRole.CLIENT, 'wait_until_done')
        return self._all_complete.wait(timeout=timeout)

    def get_task(self, timeout: float = None) -> Tuple[str, Result]:
        """Get a task object

        Args:
            timeout (float): Timeout for waiting for a task
        Returns:
            - (str) Topic of the calculation. Used in defining which queue to use to send the results
            - (Result) Task description
        Raises:
            TimeoutException: If the timeout on the queue is reached
            KillSignalException: If the queue receives a kill signal
        """
        self._check_role(QueueRole.SERVER, 'get_task')

        # Pull a record off of the queue
        topic, message = self._get_request(timeout)
        logger.debug(
            f'Received a task message with topic {topic} inbound queue')

        # Get the message
        task = Result.parse_raw(message)
        # task = Result.model_validate_json(message)
        task.mark_input_received()
        return topic, task

    def send_kill_signal(self):
        """Send the kill signal to the task server"""
        self._check_role(QueueRole.CLIENT, 'send_kill_signal')
        self._send_request("null", topic='default')

    def send_result(self, result: Result, topic: str):
        """Send a value to a client

        Args:
            result (Result): Result object to communicate back
            topic (str): Topic of the calculation
        """
        self._check_role(QueueRole.SERVER, 'send_result')
        result.mark_result_sent()
        self._send_result(result.json(), topic=topic)

    @abstractmethod
    def _get_request(self, timeout: int = None) -> Tuple[str, str]:
        """Get a task request from the client

        Args:
            timeout: Timeout for the blocking get in seconds
        Returns:
            - (str) Topic of the item
            - (str) Serialized version of the task request object
        Raises:
            (TimeoutException) if the timeout is reached
            (KillSignalException) if a kill signal (the string ``null``) is received
        """
        pass

    @abstractmethod
    def _send_request(self, message: str, topic: str):
        """Push a task request to the task server

        Args:
            message (str): JSON-serialized version of the task request object
            topic (str): Topic of the queue
        """
        pass

    @abstractmethod
    def _get_result(self, topic: str, timeout: int = None) -> str:
        """Get a result from the task server

        Returns:
            A serialized form of the result method
        Raises:
            (TimeoutException) if the timeout is reached
        """
        pass

    @abstractmethod
    def _send_result(self, message: str, topic: str):
        """Push a result object from task server to thinker

        Args:
            message (str): Serialized version of the task request object
            topic (str): Topic of the queue
        """

    @abstractmethod
    def flush(self):
        """Remove all existing results from the queues"""
        pass
