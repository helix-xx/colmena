from __future__ import annotations

import logging
import os
import redis

from typing import Any, Optional, Union

from colmena import value_server
from colmena.models import SerializationMethod


logger = logging.getLogger(__name__)

VALUE_SERVER_HOST_ENV_VAR = 'COLMENA_VALUE_SERVER_HOST'
VALUE_SERVER_PORT_ENV_VAR = 'COLMENA_VALUE_SERVER_PORT'


class ValueServer:
    """Wrapper around a Redis Client for interacting with the value server"""
    def __init__(self, hostname: str, port: int):
        """
        Args:
            hostname (str): hostname of Redis server
            port (int): port of Redis server
        """
        self.redis_client = redis.StrictRedis(
            host=hostname, port=port, decode_responses=True)

    def exists(self, key: str) -> bool:
        """Check if key exists

        Args:
            key (str)

        Returns:
            bool
        """
        return self.redis_client.exists(key)

    def get(self,
            key: str,
            serialization_method: Union[str, SerializationMethod] = SerializationMethod.PICKLE
    ) -> Optional[object]:
        """Get object by key from value server

        Args:
            key (str)
            serialization_method (SerializationMethod): serialization method
                to use for deserializing object

        Returns:
            deserialized object corresponding to key or None if key does not
            exist
        """
        value = self.redis_client.get(key)
        if value is not None:
            return SerializationMethod.deserialize(serialization_method, value)
        return None

    def put(self,
            key: str,
            obj: Any,
            serialization_method: Union[str, SerializationMethod] = SerializationMethod.PICKLE
    ) -> None:
        """Put object in value server

        Args:
            key (str)
            obj (object)
            serialization_method (SerializationMethod): serialization method
                to use for serializing object before putting in value server
        """
        value = SerializationMethod.serialize(serialization_method, obj)
        self.redis_client.set(key, value)


def init_value_server(hostname: Optional[str] = None,
                      port: Optional[int] = None) -> None:
    """Attempt to establish a Redis client connection to the value server

    Attempt to initialize the global variable `server` to a `ValueServer`
    instance using the Redis server hostname and port that are provided as
    arguments or via the environment variables defined by
    `VALUE_SERVER_HOST_ENV_VAR` and `VALUE_SERVER_PORT_ENV_VAR`. If the hostname
    and port are not provided via arguments or environment variables, we
    assume the value server is not being used.

    Args:
        hostname (str): optional Redis server hostname for the value server
        port (int): optional Redis server port for the value server
    """
    #global value_server.server

    if value_server.server is not None:
        return

    if hostname is None:
        if VALUE_SERVER_HOST_ENV_VAR in os.environ:
            hostname = os.environ.get(VALUE_SERVER_HOST_ENV_VAR)
        else:
            raise ValueError('hostname was not passed to init_value_server ',
                    'and the {} environment variable is not set'.format(
                    VALUE_SERVER_HOST_ENV_VAR))

    if port is None:
        if VALUE_SERVER_PORT_ENV_VAR in os.environ:
            port = int(os.environ.get(VALUE_SERVER_PORT_ENV_VAR))
        else:
            raise ValueError('port was not passed to init_value_server ',
                    'and the {} environment variable is not set'.format(
                    VALUE_SERVER_PORT_ENV_VAR))

    value_server.server = ValueServer(hostname, port)
