from ._library import *
from ._library import __version__, __api_version__

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

# gets sphinx autodoc done right - don't remove it
__all__ = [k for k in dir() if not k.startswith('_')]
del k
