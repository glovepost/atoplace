"""
RPC Protocol Definition

Defines the message format for communication between the MCP server (Python 3.10+)
and the KiCad worker (Python 3.9).
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json

@dataclass
class RpcRequest:
    id: str
    method: str
    params: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> 'RpcRequest':
        d = json.loads(data)
        return cls(**d)

@dataclass
class RpcResponse:
    id: str
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> 'RpcResponse':
        d = json.loads(data)
        return cls(**d)
