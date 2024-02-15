from enum import Enum

def edge_tensor_to_list(edge_tensor):
    edge_list = []
    for i in range(edge_tensor.shape[1]):
        edge_list.append((int(edge_tensor[0, i]), int(edge_tensor[1, i])))

    return edge_list

def list_of_dictionary_to_dictionary_of_lists(ls):
    return {k: [dic[k] for dic in ls] for k in ls[0]}

def dictionary_of_lists_to_list_of_dictionaries(dict):
    return [dict(zip(dict,t)) for t in zip(*dict.values())]

class dotdict(dict):
    """dot.notation access to dictionary attributes
    Source: https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def transform_dict_to_args_list(dictionary):
    """
    Transform dict to list of args
    """
    list_args = []
    for key,value in dictionary.items():
        list_args += [key, str(value)]
    return list_args

class PredictionType(Enum):
    GRAPH_PREDICTION = 1
    NODE_EMBEDDING = 2
    NODE_PREDICTION = 3