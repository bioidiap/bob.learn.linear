"""Auxiliary functions to be used for preparing the training data."""

def bic_intra_extra_pairs(training_data):
  """bic_intra_extra_pairs(training_data) -> intra_pairs, extra_pairs

  Computes intra-class and extra-class pairs from given training data.

  The ``training_data`` should be aligned in a list of sub-lists, where each sub-list contains the data of one class.
  This function will return two lists of tuples of data, where the first list contains tuples of the same class, while the second list contains tuples of different classes.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  .. note::
     In general, many more ``extra_pairs`` than ``intra_pairs`` are returned.

  .. warning::
     This function actually returns a two lists of pairs of references to the given data.
     Even for relatively low numbers of classes and elements per class, the returned lists may contain billions of pairs, which require huge amounts of memory.

  **Keyword parameters**

  training_data : [[object]]
    The training data, where the data for each class are enclosed in one list.

  **Return values**

  intra_pairs : [(object, object)]
    A list of tuples of data, where both data belong to the same class, where each data element is a reference to one element of the given ``training_data``.

  extra_pairs : [(object, object)]
    A list of tuples of data, where both data belong to different classes, where each data element is a reference to one element of the given ``training_data``.
  """
  # generate intra-class pairs
  intra_pairs = [(clazz[c1], clazz[c2]) \
                  for clazz in training_data \
                  for c1 in range(len(clazz)-1) \
                  for c2 in range (c1+1, len(clazz))
                ]

  # generate extra-class pairs
  extra_pairs = [(c1, c2) \
                  for clazz1 in range(len(training_data)-1) \
                  for c1 in training_data[clazz1] \
                  for clazz2 in range(clazz1+1, len(training_data)) \
                  for c2 in training_data[clazz2]
                ]

  # return a tuple of pairs
  return (intra_pairs, extra_pairs)

def bic_intra_extra_pairs_between_factors(first_factor, second_factor):
  """bic_intra_extra_pairs_between_factors(first_factor, second_factor) -> intra_pairs, extra_pairs

  Computes intra-class and extra-class pairs from given training data, where only pairs between the first and second factors are considered.

  Both ``first_factor`` and ``second_factor`` should be aligned in a list of sub-lists, where corresponding sub-list contains the data of one class.
  Both lists need to contain the same classes in the same order; empty classes (empty lists) are allowed.
  This function will return two lists of tuples of data, where the first list contains tuples of the same class, while the second list contains tuples of different classes.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  .. note::
     In general, many more ``extra_pairs`` than ``intra_pairs`` are returned.

  .. warning::
     This function actually returns a two lists of pairs of references to the given data.
     Even for relatively low numbers of classes and elements per class, the returned lists may contain billions of pairs, which require huge amounts of memory.

  **Keyword parameters**

  first_factor : [[object]]
    The training data for the first factor, where the data for each class are enclosed in one list.

  second_factor : [[object]]
    The training data for the second factor, where the data for each class are enclosed in one list.
    Must have the same size as ``first_factor``.

  **Return values**

  intra_pairs : [(array_like, array_like)]
    A list of tuples of data, where both data belong to the same class, but different factors.

  extra_pairs : [(array_like, array_like)]
    A list of tuples of data, where both data belong to different classes and different factors.
  """

  assert len(first_factor) == len(second_factor), "The data for both factors must contain the same number of classes"

  # generate intra-class pairs
  intra_pairs = [(c1,c2) \
                  for clazz in range(len(first_factor)) \
                  for c1 in first_factor[clazz] \
                  for c2 in second_factor[clazz]
                ]

  # generate extra-class pairs
  extra_pairs = [(c1, c2) \
                  for clazz1 in range(len(first_factor)) \
                  for c1 in first_factor[clazz1] \
                  for clazz2 in range(len(second_factor)) \
                  for c2 in second_factor[clazz2] \
                  if clazz1 != clazz2
                ]

  # return a tuple of pairs
  return (intra_pairs, extra_pairs)
