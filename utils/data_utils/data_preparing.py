import json
import numpy as np
from .pose_parameters import get_pose_params_from_mat, get_pt2d_from_mat

def get_IDs(data, list_datasets=['300W_LP', 'AFLW2000']):
    dictionary = {}
    dictionary['train'] = []
    dictionary['valid'] = []
    
    for index, row in data.iterrows():
        if row['type'] == 'train' and row['dataset'] in list_datasets:
            dictionary['train'].append(row['image'].split('.')[0])
        if row['type'] == 'val' and row['dataset'] in list_datasets:
            dictionary['valid'].append(row['image'].split('.')[0])
    return dictionary

def get_labels(dictionary):
    labels = {}
    print('Start Parsing train files')
    for idx in dictionary['train']:
        labels[idx] = get_pose_params_from_mat('Datasets/'+idx)
    print('Start Parsing valid files')
    for idx in dictionary['valid']:
        labels[idx] = get_pose_params_from_mat('Datasets/'+idx)
    return labels
    
def get_pt2d(dictionary):
    pt2d = {}
    print('Start Parsing train files')
    for idx in dictionary['train']:
        pt2d[idx] = get_pt2d_from_mat('Datasets/' + idx)
    print('Start Parsing valid files')
    for idx in dictionary['valid']:
        pt2d[idx] = get_pt2d_from_mat('Datasets/' + idx)
    return pt2d

def dictionary_to_json(dictionary, file_name):
    partition = json.dumps(dictionary, cls=NumpyEncoder)
    with open('Datasets/' + file_name + '.json', 'w') as f:
        f.write(partition + '\n')

def load_json(json_file_path):
    with open(json_file_path, 'r') as j:
        data = json.loads(j.read())
        j.close()
    return data

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)