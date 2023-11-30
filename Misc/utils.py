from collections import deque
from itertools import chain
from sys import getsizeof, stderr

import networkx as nx


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

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def num_connected_components(data):
    G = nx.Graph()

    edge_list = edge_tensor_to_list(data.edge_index)

    for i, edge in enumerate(edge_list):
        G.add_edge(edge[0], edge[1])
    return nx.number_connected_components(G)