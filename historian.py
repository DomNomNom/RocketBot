import ctypes
from io import BytesIO
import base64

import game_data_struct
import bot_input_struct


# hack the ctypes.Structure class to include printing the fields
class Struct(ctypes.Structure):
    def __repr__(self):
        '''Print the fields'''
        res = []
        for field in self._fields_:
            res.append('%s=%s' % (field[0], repr(getattr(self, field[0]))))
        return self.__class__.__name__ + '(' + ','.join(res) + ')'
    @classmethod
    def from_param(cls, obj):
        '''Magically construct from a tuple'''
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, tuple):
            return cls(*obj)
        raise TypeError


class HistoryItem(Struct):
    _fields_ = [
        ('time', ctypes.c_float),
        ('game_tick_packet', game_data_struct.GameTickPacket),
        ('output_vector', bot_input_struct.PlayerInput),
    ]

    # https://stackoverflow.com/questions/7021841/get-the-binary-representation-of-a-ctypes-structure-in-python
    def encode(self):
        fakefile = BytesIO()
        fakefile.write(self)
        return base64.b64encode(fakefile.getvalue())
    @classmethod
    def decode(cls, line):
        fakefile = BytesIO(my_encoded_c_struct)
        history_item = HistoryItem()
        fakefile.readinto(history_item)
        return history_item

def ctype_equal(obj1, obj2):
    for fld in obj1._fields_:
        if getattr(obj1, fld[0]) != getattr(obj2, fld[0]):
            return False
    return True

# time -> output_vector
def to_action_dict(history):
    trimmed = [
        item for i, item in enumerate(history)
        if (not i) or (not ctype_equal(history[i-1].output_vector, item.output_vector)) ]
    return { item.time: item.output_vector for item in trimmed }

