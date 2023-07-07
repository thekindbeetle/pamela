import numpy as np
import scipy.io

from calorimeter.calorimeter import Calorimeter

# Список плоскостей
planes = list(range(1, 45))
planes.remove(38)

full_key_list = ['U{0}'.format(str(idx)) for idx in planes]
full_keys = dict(zip(planes, full_key_list))

key_list = ['U{0}'.format(str(idx)) for idx in planes]
keys = dict(zip(planes, key_list))


def parse_event(data, evnum):

    ev = np.zeros((44, 96))
    for plnum in planes:
        for i in range(96):
            ev[plnum - 1][i] = data[plnum][evnum][i]
    return ev


def import_data(fname, events=None):
    """
    Импорт данных по калориметру из mat-файла.
    :param: fname Путь к файлу
    :param: events Диапазон событий
    """
    mat_data = scipy.io.loadmat(fname, variable_names=key_list)
    print('.mat file loaded')

    data = dict()
    mat_shape = (0, 96)
    for idx in full_keys.keys():
        pl_data = mat_data.get(full_keys[idx])
        if scipy.sparse.issparse(pl_data):
            data[idx] = pl_data.toarray()
            mat_shape = pl_data.shape
        else:
            data[idx] = np.zeros(mat_shape)
    if events is None:
        start_event = 1
        last_event = len(data[1])
    else:
        start_event, last_event = events[0], events[1]
    return dict([(e, Calorimeter(parse_event(data, e - 1))) for e in range(start_event, last_event + 1)])
