import pynvml
import struct
from dataclasses import dataclass

@dataclass
class Packet:
    node_id: int
    locak_rank: int
    seq: int
    metric_type: int  # TODO: add enum
    timestamp: float  
    value: float

    FMT = "!B I I H d d"  # 27 bytes

    def pack(self) -> bytes:
        return struct.pack(self.FMT,
            self.version, self.node_id, self.seq,
            self.metric_type, self.timestamp, self.value)

    @classmethod
    def unpack(cls, data: bytes) -> "Packet":
        return cls(*struct.unpack(cls.FMT, data))


class Sender():
    def __init__(self, node_id, dest_addr, dest_port, generator):
        self.node_id = node_id
        self.dest_addr = dest_addr
        self.dest_port = dest_port
        # yield data from training loop
        self.generator = generator
    
    def connection_made(self):
        pass

    async def send_udp(self):
        pass



