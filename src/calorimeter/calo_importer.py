import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from calorimeter.calorimeter import Calorimeter

# Список плоскостей
planes = list(range(1, 45))
planes.remove(38)

full_key_list = ['U{0}'.format(str(idx)) for idx in planes]
full_keys = dict(zip(planes, full_key_list))

key_list = ['U{0}'.format(str(idx)) for idx in planes]
keys = dict(zip(planes, key_list))


def import_data(fname, events=None, verbose=True):
    """
    Импорт данных по калориметру из mat-файла.
    :param: fname Путь к файлу
    :param: events Номера событий
    """
    mat_data = loadmat(fname, variable_names=key_list)
    if verbose:
        print('.mat file loaded')

    size = mat_data['U1'].shape[0]
    data = np.zeros((size, 44, 96))
    for idx in full_keys.keys():
        pl_data = mat_data.get(full_keys[idx])
        if issparse(pl_data):
            data[:, idx - 1, :] = pl_data.toarray()
    if events is None:
        events = range(1, size + 1)
    return dict([(e, Calorimeter(data[e - 1, :, :])) for e in events])
