
class Instance(object):

  def __init__(self, instance_dict, instance_idx=None):

    self.instance_idx = instance_idx
    self.endogenous_nodes_dict = dict()
    self.exogenous_nodes_dict = dict()
    for key, value in instance_dict.items():
      if 'x' in key:
        self.endogenous_nodes_dict[key] = value
      elif 'u' in key:
        self.exogenous_nodes_dict[key] = value
      else:
        raise Exception(f'Node type not recognized.')

  def dict(self, node_types = 'endogenous'):
    if node_types == 'endogenous':
      return self.endogenous_nodes_dict
    elif node_types == 'exogenous':
      return self.exogenous_nodes_dict
    elif node_types == 'endogenous_and_exogenous':
      return {**self.endogenous_nodes_dict, **self.exogenous_nodes_dict}
    else:
      raise Exception(f'Node type not recognized.')

  # def array(self, node_types = 'endogenous'): #, nested = False
  #   return np.array(
  #     list( # TODO (BUG???) what happens to this order? are values always ordered correctly?
  #       self.dict(node_types).values()
  #     )
  #   ).reshape(1,-1)

  def keys(self, node_types = 'endogenous'):
    return self.dict(node_types).keys()

  def values(self, node_types = 'endogenous'):
    return self.dict(node_types).values()

  def items(self, node_types = 'endogenous'):
    return self.dict(node_types).items()