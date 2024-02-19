import ctypes


def fill_struct(instance, *params):
    idx = 0
    for field in instance._fields_:
        if issubclass(field[1], ctypes.Structure):
            idx += fill_struct(instance.__getattribute__(field[0]), *params[idx:])
        else:
            instance.__setattr__(field[0], params[idx])
            idx += 1

    return idx


def make_struct(struct_t, *args):
    p = struct_t()
    fill_struct(p, *args)
    return p
