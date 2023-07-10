"""
Поиск трека антипротона в калориметре
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import tqdm

from calorimeter import calo_importer
from calorimeter.find_apr_track import get_start_direction, get_star

disk = 'E'

# Номера событий
# Нумерация сквозная, в Event потом пересчитывается
# Т.е. берём диапазон событий от минимума до максимума, строим индекс (есть в Event или нет)
# и для триггерных событий применяем этот индекс
nn = range(4, 8)
n_min, n_max = min(nn), max(nn)

fnum = 1
dpath = '{0}:/YandexDisk/data/2023/antiprotons/'.format(disk)
dfile = 'CaloStrip/CaloStrip_apr_0p1_1500_rig_F0p8_10000000_{fnum}.mat'.format(fnum=fnum)
l2file = 'CaloL2/CaloL2_apr_0p1_1500_rig_F0p8_10000000_{fnum}.mat'.format(fnum=fnum)
ntrackfile = 'Ntrack/Ntrack_apr_0p1_1500_rig_F0p8_10000000_{fnum}.mat'.format(fnum=fnum)
trackerfile = 'Tracker/Tracker_apr_0p1_1500_rig_F0p8_10000000_{0}.mat'.format(fnum)
siminfofile = 'SimInfo/SimInfo_apr_0p1_1500_rig_F0p8_10000000_{0}.mat'.format(fnum)

# dpath = '{0}:/YandexDisk/data/2023/protons/'.format(disk)
# dfile = 'CaloStrip/CaloStrip_pr_0p1_1500_rig_F0p8_10000000_1.mat'
# l2file = 'CaloL2/CaloL2_pr_0p1_1500_rig_F0p8_10000000_1.mat'
# ntrackfile = 'Ntrack/Ntrack_pr_0p1_1500_rig_F0p8_10000000_1.mat'

print('Loading events...')
calo_events = calo_importer.import_data(os.path.join(dpath, dfile), events=nn)

# Номера событий
event_data = scipy.io.loadmat(os.path.join(dpath, ntrackfile), variable_names=['Event'])['Event'].flatten()
event_idx = [event_data[i] in nn for i in range(len(event_data))]
event_numbers = event_data[event_idx]

calo_events = dict([(i, calo_events[i]) for i in event_numbers])

tibar_data = scipy.io.loadmat(os.path.join(dpath, l2file), variable_names=['tibarX', 'tibarY'])  # Стартовые точки
tibarX, tibarY = tibar_data['tibarX'][event_idx, :], tibar_data['tibarY'][event_idx, :]
start_pts = np.vstack([tibarX[:, 0], tibarY[:, 0]]).transpose()
start_pts = dict([(event_numbers[i], start_pts[i]) for i in range(len(event_numbers))])
tibarX = dict([(event_numbers[i], tibarX[i]) for i in range(len(event_numbers))])
tibarY = dict([(event_numbers[i], tibarY[i]) for i in range(len(event_numbers))])

rig_data = scipy.io.loadmat(os.path.join(dpath, trackerfile), variable_names=['Rig'])['Rig'].flatten()
siminfo_data = scipy.io.loadmat(os.path.join(dpath, siminfofile), variable_names=['caloplanepos', 'caloplaneint', 'fTHETA', 'fPHI'])
calol2_data = scipy.io.loadmat(os.path.join(dpath, l2file), variable_names=['nstrip'])

# Жесткость частицы
rig = dict([(event_numbers[i], rig_data[i]) for i in range(len(event_numbers))])
# Плоскость взаимодействия
caloplaneint = dict([(event_numbers[i], siminfo_data['caloplaneint'][event_numbers[i] - 1][0]) for i in range(len(event_numbers))])
# Количество сработавших стрипов
nstrip = dict([(event_numbers[i], int(calol2_data['nstrip'][event_numbers[i] - 1])) for i in range(len(event_numbers))])

# Зенитный (полярный) угол влета частицы
ev_theta = dict([(event_numbers[i], siminfo_data['fTHETA'][event_numbers[i] - 1][0]) for i in range(len(event_numbers))])
ev_phi = dict([(event_numbers[i], siminfo_data['fPHI'][event_numbers[i] - 1][0]) for i in range(len(event_numbers))])
print('Load completed.')


# for evnum in tqdm.tqdm(event_numbers):
for evnum in event_numbers:
    # print('Rig = {0}, nstrip = {1}, caloplaneint = {2}'.format(rig[evnum], nstrip[evnum], caloplaneint[evnum]))

    img_x = calo_events[evnum].x_projection()
    img_y = calo_events[evnum].y_projection()

    start_x, start_y, full_angle, start_angle_x, start_angle_y = \
        get_start_direction(img_x, img_y,
                            weight_power=-1.8, shift=0.5, plot_track=False, num_directions=200,
                            plot_title='Event {0}'.format(evnum),
                            verbose=False)

    print("Event {0}: ({1}, {2}) vs real ({3}, {4})".format(evnum, start_x + 1, start_y + 1,
                                                            start_pts[evnum][0], start_pts[evnum][1]))

    get_star(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y,
             peak_threshold=8.0, peak_min_difference=np.pi / 12, weight_power=-0.3,
             weight_shift=0.25, default_sigma=0.05, max_peaks=100,
             verbose=False, plot=False, plot_result=True, output_file=None)

    plt.show()
