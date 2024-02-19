from utils.tools import make_struct

import ctypes
import socket

from typing import List, Iterable, ClassVar


def connect(ip: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    return sock


def recv_exact_n_bytes_into(sock: socket.socket, n: int, buf: memoryview):
    fetch_max = 1024
    cur_bytes = 0

    while cur_bytes < n:
        to_fetch = fetch_max if n - cur_bytes > fetch_max else n - cur_bytes
        cur_bytes += sock.recv_into(buf[cur_bytes:], to_fetch)


def struct_pack_into(pack_struct: ClassVar, to_pack: Iterable, buf: ctypes.Array[ctypes.c_char]):
    pack_instance = make_struct(pack_struct, *to_pack)
    ctypes.memmove(buf, ctypes.addressof(pack_instance), ctypes.sizeof(pack_instance))


def struct_unpack_from(unpack_struct: ClassVar, buf: ctypes.Array[ctypes.c_char]) -> List:
    unpack_instance = unpack_struct()
    ctypes.memmove(ctypes.addressof(unpack_instance), buf, ctypes.sizeof(unpack_instance))

    ret = []
    for field in unpack_instance._fields_:
        content = unpack_instance.__getattribute__(field[0])
        if isinstance(content, ctypes.Array):
            ret.extend(content)
        else:
            ret.append(content)

    return ret
