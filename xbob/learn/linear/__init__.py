from ._library import *
from ._library import __version__, __api_version__
from xbob.learn.activation import Identity

def machine_repr(self):
  """A funky way to display a bob Linear Machine"""
  if self.activation == Identity():
    return '<%s.Machine %s@%s>' % (__name__, self.weights.dtype, self.weights.shape)
  else:
    return '<%s.Machine %s@%s [act: %s]>' % (__name__, self.weights.dtype, self.weights.shape, self.activation)
Machine.__repr__ = machine_repr
del machine_repr

def machine_str(self):
  """A funky way to print a bob Linear Machine"""
  act = ""
  if self.activation != Identity():
    act = " [act: %s]" % self.activation
  sub = ""
  if not (self.input_subtract == 0.0).all():
    sub = "\n subtract: %s" % self.input_subtract
  div = ""
  if not (self.input_divide == 1.0).all():
    div = "\n divide: %s" % self.input_divide
  bias = ""
  if not (self.biases == 0.0).all():
    bias = "\n bias: %s" % self.biases
  shape = self.weights.shape
  return '%s.Machine (%s) %d inputs, %d outputs%s%s%s%s\n %s' % \
      (__name__, self.weights.dtype, shape[0], shape[1], act, sub, div,
          bias, self.weights)
Machine.__str__ = machine_str
del machine_str

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')
