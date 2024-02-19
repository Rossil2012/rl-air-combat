import ctypes
import numpy as np
from enum import Enum


PI: float = 3.14159265358979323846
CUR_EPOCH_DATA_ID: int = 0
INIT_CALLBACK_ID: int = 400000
PROCESS_CALLBACK_ID: int = 400001
ENDGAME_CALLBACK_ID: int = 400002
RESTART_CALLBACK_ID: int = 400003
EXIT_CALLBACK_ID: int = 400004

HEALTH_MAX = 100.
DAMAGE_MAX = 5.
DAMAGE_DIST_MAX = 800.
DAMAGE_ANGLE_MAX = np.pi / 15.

ACTION_INFO = 4, [-1e3, -1e3, -1e3, 0.], [1e3, 1e3, 1e3, 1e3]
NO_GUN_OBS_INFO = 12 * 2, \
    [-1e5, -1e5, 0, -PI / 4., -PI / 3., 0., -1e3, -1e3, -1e3, -PI / 4, -PI / 4, -PI / 4], \
    [1e5, 1e5, 1e5, PI / 4., PI / 3., PI * 2., 1e3, 1e3, 1e3, PI / 4, PI / 4, PI / 4]
COMBAT_OBS_INFO = 14 * 2, \
    [-1e5, -1e5, 0, -PI / 4., -PI / 3., 0., -1e3, -1e3, -1e3, -PI / 4, -PI / 4, -PI / 4, 0, 0] * 2, \
    [1e5, 1e5, 1e5, PI / 4., PI / 3., PI * 2., 1e3, 1e3, 1e3, PI / 4, PI / 4, PI / 4, DAMAGE_MAX, HEALTH_MAX] * 2


class DaotiaoType(Enum):
    EXIT = 0
    NONE = 1
    WATCH = 2
    PAUSE = 3
    STEP_FORWARD = 4
    UPDATE_ROOM_SETTING = 5
    UPDATE_PLAYER_STATE = 6
    RESUME = 7


class OperationType(Enum):
    NONE_OPERATION = 0
    EPOCH_COMM_TO_CLIENT = 1
    EPOCH_CLIENT_TO_COMM = 2
    ROOM_SETTING = 3
    ROOM_SETTING_RESPON = 4
    CLIENT_SETTING = 5
    INI_POINT = 6
    GAME_OVER = 7
    DAOTIAO_SETTING = 8

    GET_EPOCH_DATA = 9
    SET_EPOCH_DATA = 10

    GET_ROOM_HISTORY_LIST = 11
    LOGIN = 12
    LOGOUT = 13
    GET_ROOM_HISTORY_DETAIL = 14
    GET_ROOM_DETAIL = 15
    GET_ROOM_LIST = 16
    GET_USER_DETAIL = 17
    GET_USER_LIST = 18

    EPOCH_DAOTIAO_TO_COMM = 19
    EPOCH_COMM_TO_DAOTIAO = 20

    TRAINING_RESTART = 21
    TRAINING_EXIT = 22


class FireType(Enum):
    NONE_FIRE = 0
    GUN = 1


class KineSendStruct(ctypes.Structure):
    _fields_ = [
        ('roll', ctypes.c_double),
        ('pitch', ctypes.c_double),
        ('yaw', ctypes.c_double),
        ('throttle', ctypes.c_double)
    ]


class KineRecvStruct(ctypes.Structure):
    _fields_ = [
        ('q123', ctypes.c_double * 3),
        ('q456', ctypes.c_double * 3),
        ('u123', ctypes.c_double * 3),
        ('u456', ctypes.c_double * 3)
    ]


class KineStateInitStruct(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
        ('z', ctypes.c_double),
        ('u', ctypes.c_double),
        ('v', ctypes.c_double),
        ('w', ctypes.c_double),
    ]


class WebpackHeader(ctypes.Structure):
    _fields_ = [
        ('operationType', ctypes.c_int32),
        ('daotiaoType', ctypes.c_int32),
        ('timestamp', ctypes.c_uint32),
        ('contentLength', ctypes.c_int32),
        ('callbackID', ctypes.c_int32)
    ]


class Posture(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('u', ctypes.c_float),
        ('v', ctypes.c_float),
        ('w', ctypes.c_float),
        ('vx', ctypes.c_float),
        ('vy', ctypes.c_float),
        ('vz', ctypes.c_float),
        ('vu', ctypes.c_float),
        ('vv', ctypes.c_float),
        ('vw', ctypes.c_float),
    ]


class TrainingModeInitInfo(ctypes.Structure):
    _fields_ = [
        ('flightID', ctypes.c_int32),
        ('initPosture', Posture)
    ]


class TrainingModeProcessInfo(ctypes.Structure):
    _fields_ = [
        ('userID', ctypes.c_int32),
        ('tick', ctypes.c_int32),
        ('self', Posture),
        ('m_HPNormalized', ctypes.c_float),
    ]


class FlightControl(ctypes.Structure):
    _fields_ = [
        ('pitch', ctypes.c_float),
        ('roll', ctypes.c_float),
        ('yaw', ctypes.c_float),
        ('throttle', ctypes.c_float)
    ]


class WeaponControl(ctypes.Structure):
    _fields_ = [
        ('fireType', ctypes.c_int32)
    ]
