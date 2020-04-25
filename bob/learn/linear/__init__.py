# import Libraries of other lib packages
import bob.io.base
import bob.math
import bob.learn.activation
import numpy

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.learn.linear', __file__)

from ._library import *
from . import version
from .version import module as __version__
from .version import api as __api_version__
from ._library import Machine as _Machine_C

from .auxiliary import *
from .GFK import GFKMachine, GFKTrainer

def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]


class Machine(_Machine_C):
    __doc__ = _Machine_C.__doc__

    def update_dict(self, d):
        self.input_sub = numpy.array([d["input_sub"]])
        self.input_div = numpy.array([d["input_div"]])

    @classmethod
    def create_from_dict(cls, d):        
        machine = cls(numpy.array(d["weights"]))
        machine.update_dict(d)
        return machine

    @staticmethod
    def to_dict(machine):
        machine_data = dict()
        machine_data["input_sub"] = machine.input_sub
        machine_data["input_div"] = machine.input_div
        machine_data["weights"] = machine.weights
        return machine_data

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        self.__dict__ = d        
        self.__init__(numpy.array(d["weights"]))
        self.update_dict(d)
