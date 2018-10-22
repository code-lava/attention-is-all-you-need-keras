import os
import jsonpickle

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def store_settings(store_object, json_file):
    # convert args to dict
    with open(json_file, 'w') as fobj:
        json_obj = jsonpickle.encode(store_object)
        fobj.write(json_obj)

def load_settings(json_file):
    # convert args to dict
    with open(json_file, 'r') as fobj:
        json_obj = fobj.read()
        obj = jsonpickle.decode(json_obj)

    return obj
