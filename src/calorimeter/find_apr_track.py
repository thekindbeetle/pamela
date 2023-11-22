"""
Поиск трека антипротона в калориметре
Версия 05.07.2023
Полную версию см. в блокноте 2023-07-07-va-calo-track-lines
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats
import scipy.signal
import scipy.io

from calorimeter.calo_utils import K, max_image_angle, get_angle_by_projections, plot_line_by_start_point_and_angle,\
    real_angle2image, image_angle2real

from util.coord_transform import cartesian2polar
from hough_transform.hough2d import hough_line
from skimage.transform import hough_line_peaks

matplotlib.use('QT5Agg')

# Матрица расстояний, которая передаётся в преобразование Хафа
DIST_MATRIX = np.transpose([[np.sqrt(j + 1) for j in np.arange(0, 22, 1)] for i in np.arange(0, 96, 1)])

### - - - - - - - - - ###
# Константы
### - - - - - - - - - ###

def_peak_threshold = 15.0  # peak_threshold - минимальное значение максимума
def_peak_min_difference = np.pi / 18  # peak_min_difference - минимальное угловое расстояние между пиками
def_inclination_weight_power = -1.8  # weight_power - показатель весовой функции
def_inclination_weight_shift = 0.5  # weight_shift - смещение весовой функции
def_star_weight_power = -0.3  # weight_power - показатель весовой функции
def_star_weight_shift = 0.25  # weight_shift - смещение весовой функции
def_default_sigma = 0.05  # default_sigma - ширина сглаживающей гауссианы
def_max_peaks = 100  # max_peaks - максимальное количество порожденных частиц


# Если трек проходит через клеточки, их надо удалить.
# Расстояние от точки (pointX, pointY) до прямой [(lineX1, lineY1), (lineX2, lineY2)]
def _dist_from_line_to_point(lineX1, lineY1, lineX2, lineY2, pointX, pointY):
    line_length = np.linalg.norm([lineX1 - lineX2, lineY1 - lineY2])
    u = (((pointX - lineX1) * (lineX2 - lineX1)) + ((pointY - lineY1) * (lineY2 - lineY1))) / (
            line_length * line_length)
    ix = lineX1 + u * (lineX2 - lineX1)
    iy = lineY1 + u * (lineY2 - lineY1)
    return np.linalg.norm([pointX - ix, pointY - iy])


# Если трек проходит через клеточки, их надо удалить.
# Расстояние от точки (pointX, pointY) до отрезка [(lineX1, lineY1), (lineX2, lineY2)]
def _dist_from_segment_to_point(lineX1, lineY1, lineX2, lineY2, pointX, pointY):
    line_length = np.linalg.norm([lineX1 - lineX2, lineY1 - lineY2])

    u = (((pointX - lineX1) * (lineX2 - lineX1)) + ((pointY - lineY1) * (lineY2 - lineY1))) / (
            line_length * line_length)

    if (u > 1) or (u < 0):
        # Ближайший конец отрезка
        return min(np.linalg.norm([lineX1 - pointX, lineY1 - pointY]),
                   np.linalg.norm([lineX2 - pointX, lineY2 - pointY]))
    else:
        # Ближайшая точка отрезка
        ix = lineX1 + u * (lineX2 - lineX1)
        iy = lineY1 + u * (lineY2 - lineY1)
        return np.linalg.norm([pointX - ix, pointY - iy])


def _gaussian_sum_1d(val, centers, sigma):
    """
    Сумма гауссиан с указанными центрами и фиксированной шириной
    :param val: точки, в которых считается распределение
    :param centers: центры гауссиан
    :param sigma: сигмы гауссиан
    :return: значения распределения в точках val
    """
    field = np.zeros(val.shape)
    for center in centers:
        mv = scipy.stats.norm(center, sigma)
        field += mv.pdf(val)
    return field


def _gaussian_sum_1d_weighted(val, centers, sigma, weights):
    """
    Сумма гауссиан с указанными центрами и фиксированной шириной и различными весами
    :param val: точки, в которых считается распределение
    :param centers: центры гауссиан
    :param sigma: сигмы гауссиан
    :param weights: веса, с которыми берутся гауссианы
    :return: значения распределения в точках val
    """
    field = np.zeros(val.shape)
    for i in range(len(centers)):
        center = centers[i]
        mv = scipy.stats.norm(center, sigma)
        field += mv.pdf(val) * weights[i]
    return field


def _gaussian_sum_1d_weighted_circular(val, centers, sigma, weights):
    """
    Сумма гауссиан с указанными центрами и фиксированной шириной и различными весами
    TODO: сделать аккуратнее
    :param val: точки, в которых считается распределение (здесь мы считаем, что они равномерно распределены от -pi до pi!
    :param centers: центры гауссиан
    :param sigma: сигмы гауссиан
    :param weights: веса, с которыми берутся гауссианы
    :return: значения распределения в точках val
    """
    field = np.zeros(val.shape)
    for i in range(len(centers)):
        center = centers[i]
        mv = scipy.stats.norm(center, sigma)
        field += mv.pdf(val) * weights[i]
        field += mv.pdf(val - np.pi * 2) * weights[i]
        field += mv.pdf(val + np.pi * 2) * weights[i]
    return field


def get_start_direction(img_x_src, img_y_src, weight_power=-1.8, shift=0.25, num_directions=80,
                        max_theta=max_image_angle,
                        plot_track=False, plot_title='', verbose=False):
    """
    Вычисляем направление влёта частицы в калориметр.
    :param img_x_src: Проекция X
    :param img_y_src: Проекция Y
    :param weight_power: Весовой коэффициент в матрице преобразования Хафа
    :param shift: Сдвиг для весовой функции
    :param max_theta: Ограничение по углу влёта
    :param plot_track: Показать картинку с треком
    :param plot_title: Заголовок для графика
    :param verbose: Включить текстовый вывод
    :param num_directions: Количество различных углов (для преобразования Хафа)
    :return: X, Y, угол влёта, углы влёта в проекциях X, Y
    """
    vprint = print if verbose else lambda *a, **k: None

    # Диапазон допустимых питч-углов
    pitch_theta = np.linspace(-max_theta, max_theta, num_directions, endpoint=False)

    img_x = (img_x_src > 0).astype(np.float64) * np.power(DIST_MATRIX + shift, weight_power)
    img_y = (img_y_src > 0).astype(np.float64) * np.power(DIST_MATRIX + shift, weight_power)

    hX, thetaX, dX = hough_line(img_x, theta=pitch_theta)
    hY, thetaY, dY = hough_line(img_y, theta=pitch_theta)

    # Выбираем стартовую точку
    # Есть проблема: восстановленный угол может быть больше максимально возможного угла.
    # Здесь фактически выбирается допустимое направление с наибольшим значением суммы аккумуляторов Хафа
    ix, iy = 0, 0
    peaksX = hough_line_peaks(hX, thetaX, dX, threshold=0.0)
    peaksY = hough_line_peaks(hY, thetaY, dY, threshold=0.0)

    ix_max = len(peaksX[0]) - 1
    iy_max = len(peaksY[0]) - 1

    vprint('Peaks X: {0}'.format(peaksX))
    vprint('Peaks Y: {0}'.format(peaksY))

    while (ix <= ix_max) and (iy <= iy_max):
        vprint('Check ix = {0}, iy = {1}'.format(ix, iy))
        vprint('Functions sum = {0:.3f}'.format(peaksX[0][ix] + peaksY[0][iy]))

        full_angle = get_angle_by_projections(peaksX[1][ix], peaksY[1][iy])

        vprint('Full angle = {0:.3f}'.format(full_angle))

        if full_angle > max_theta:
            if ix == ix_max:
                iy += 1
            elif iy == iy_max:
                ix += 1
            elif (peaksX[0][ix] - peaksX[0][ix + 1]) <= (peaksY[0][iy] - peaksY[0][iy + 1]):
                ix += 1
            else:
                iy += 1
        else:
            break

        if (ix > ix_max) or (iy > iy_max):
            ix, iy = 0, 0
            break

    r_max_x, theta_max_x = peaksX[2][ix], peaksX[1][ix]
    r_max_y, theta_max_y = peaksY[2][iy], peaksY[1][iy]

    start_x = r_max_x / np.cos(theta_max_x)
    start_y = r_max_y / np.cos(theta_max_y)

    # Корректируем угол влёта с учётом разного масштаба по осям X/Y и Z
    start_angle_x_corr = image_angle2real(theta_max_x)
    start_angle_y_corr = image_angle2real(theta_max_y)

    # Восстанавливаем угол влёта по двум проекциям
    real_full_angle = get_angle_by_projections(start_angle_x_corr, start_angle_y_corr)

    if plot_track:
        fig, ax = plt.subplots(2, 1)
        fig.suptitle(plot_title)
        fig.set_figwidth(12)
        fig.set_figheight(5)

        ax0 = ax[0].imshow(img_x_src, norm=colors.LogNorm(vmin=0.7, vmax=10.0), cmap='Greys')
        ax[0].axline((start_x, 0), slope=np.tan(theta_max_x + np.pi / 2), color='r', lw=2)
        # fig.colorbar(ax0, ax=ax[0], extend='max')

        ax1 = ax[1].imshow(img_y_src, norm=colors.LogNorm(vmin=0.7, vmax=10.0), cmap='Greys')
        ax[1].axline((start_y, 0), slope=np.tan(theta_max_y + np.pi / 2), color='r', lw=2)
        # fig.colorbar(ax1, ax=ax[1], extend='max')

        fig.colorbar(ax0, ax=ax.ravel().tolist())
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.175, 0.03, 0.65])
        # fig.colorbar(ax0, cax=cbar_ax)

    return start_x, start_y, real_full_angle, theta_max_x, theta_max_y
    # return start_x, start_y, real_full_angle, start_angle_x_corr, start_angle_y_corr


def get_track_dedx(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y, radius=1):
    """
    Энерговыделения в треке частицы
    :param img_x: проекция X
    :param img_y: проекция Y
    :param start_x: точка влёта в проекцию X
    :param start_y: точка влёта в проекцию Y
    :param start_angle_x: угол влёта в проекцию X (на картинке)
    :param start_angle_y: угол влёта в проекцию Y (на картинке)
    :param radius: количество стрипов, используемых для определения энерговыделения вдоль трека (в обе стороны)
        (лучшее значение = 1, т.е. три стрипа)
    @return:
    """
    x_track = np.round(start_x - np.tan(start_angle_x) * np.arange(0.5, 22.5, 1.0)).astype(int)
    y_track = np.round(start_y - np.tan(start_angle_y) * np.arange(0.5, 22.5, 1.0)).astype(int)

    # Если трек вылетел за пределы калориметра, ставим 0;
    # Если нет - берём стрип и два соседних
    dedx_track_x = []
    for z in range(22):
        if (x_track[z] < radius) or (x_track[z] > 95 - radius):
            dedx_track_x += [0.0]
        else:
            dedx_track_x += [sum([img_x[z, x] for x in range(x_track[z] - radius, x_track[z] + radius + 1)])]

    # Если трек вылетел за пределы калориметра, ставим 0;
    # Если нет - берём стрип и radius * 2 соседних
    dedx_track_y = []
    for z in range(22):
        if (y_track[z] < radius) or (y_track[z] > 95 - radius):
            dedx_track_y += [0.0]
        else:
            dedx_track_y += [sum([img_y[z, y] for y in range(y_track[z] - radius, y_track[z] + radius + 1)])]

    return dedx_track_x, dedx_track_y


def get_max_dedx_track_positions(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y, radius=1,
                                 num=3, plot=False):
    """
    Максимум энерговыделений вдоль трека.
    В среднем должен соответствовать плоскости взаимодействия.
    :param img_x: проекция X
    :param img_y: проекция Y
    :param start_x: точка влёта в проекцию X
    :param start_y: точка влёта в проекцию Y
    :param start_angle_x: угол влёта в проекцию X (на картинке)
    :param start_angle_y: угол влёта в проекцию Y (на картинке)
    :param num: количество максимумов энерговыделения (возможно, меньше)
    :param radius: количество стрипов, используемых для определения энерговыделения вдоль трека (в обе стороны)
        (лучшее значение = 1, т.е. три стрипа)
    :param plot: рисовать график энерговыделений
    :return: Список пиков, список энерговыделений вдоль трека в проекциях X, Y
    """
    dedx_track_x, dedx_track_y = \
        get_track_dedx(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y, radius=radius)

    # TODO: нужен ли здесь параметр height?
    # peaks_info = scipy.signal.find_peaks(np.array(dedx_track_x) + np.array(dedx_track_y), height=4.0, distance=1)
    peaks_info = scipy.signal.find_peaks(np.array(dedx_track_x) + np.array(dedx_track_y), height=0.0, distance=1)

    # Сортируем в порядке убывания значений
    if len(peaks_info[0]) > 0:
        peaks_sorted = peaks_info[0][np.argsort(peaks_info[1]['peak_heights'])][::-1]
    else:
        peaks_sorted = []

    if plot:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.arange(1, 23, 1), dedx_track_x)
        ax[0].set_title('X')
        ax[1].plot(np.arange(1, 23, 1), dedx_track_y)
        ax[0].set_title('Y')
        for peak in peaks_sorted:
            ax[0].plot([peak + 1, peak + 1], [0, dedx_track_x[peak]], '--r')
            ax[1].plot([peak + 1, peak + 1], [0, dedx_track_y[peak]], '--r')

    return peaks_sorted[:num], np.array(dedx_track_x), np.array(dedx_track_y)


def get_star(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y,
             peak_threshold=0.0, peak_min_difference=np.pi / 18, weight_power=-0.5,
             weight_shift=0.25, default_sigma=0.05, max_peaks=100, search_max=True, num_max_positions=3,
             track_radius=1, method='default',
             filter_img=None, verbose=False, plot=False, plot_result=False, output_file=None):
    """
    Поиск "звезды" взаимодействия.
    :param img_x: проекция X калориметра
    :param img_y: проекция Y калориметра
    :param start_x: точка влёта в калориметр (координата X)
    :param start_y: точка влёта в калориметр (координата Y)
    :param start_angle_x: угол влёта в калориметр (координата X), на картинке (!)
    :param start_angle_y: угол влёта в калориметр (координата Y), на картинке (!)
    :param peak_threshold: минимальное значение функции для признания точки максимума пиком
    :param peak_min_difference: минимальное угловое расстояние между пиками
    :param weight_power: показатель весовой функции (для поиска звезды)
    :param weight_shift: смещение весовой функции
    :param default_sigma: ширина сглаживающей гауссианы
    :param max_peaks: максимальное количество порожденных частиц
    :param search_max: перебирать только максимумы энерговыделений вдоль трека (иначе проходим по всем точкам)
    :param num_max_positions: количество исследуемых максимумов
    :param track_radius: ширина цилиндра вдоль трека, по которой ищутся максимумы энерговыделения
    :param method: метод сглаживания распределения в полярных координатах ('default', 'kde')
    :param filter_img: функция фильтрации изображения перед поиском точки взаимодействия
    :param verbose: включить текстовый вывод
    :param plot: нарисовать графики
    :param plot_result: вывести конечную звезду
    :param output_file: сохранить изображение в файл
    :return: В возвращаемом результате значения масштабированы!
    """
    vprint = print if verbose else lambda *a, **k: None

    polar_step = np.pi / 360
    polar_angles_list = np.arange(-np.pi, np.pi, polar_step)
    peak_angle_difference = (peak_min_difference // polar_step) + 1

    img_xf, img_yf = np.copy(img_x), np.copy(img_y)

    if filter_img is not None:
        img_xf = filter_img(img_x)
        img_yf = filter_img(img_y)

    zx, x = np.where(img_xf > 0)
    zy, y = np.where(img_yf > 0)
    zx = zx / K  # Масштабируем ось Z
    zy = zy / K

    # Нижний радиус используется для того, чтобы убрать точки с трека первичной частицы
    radius_lower = np.sqrt(2)
    # Верхний радиус используется для того, чтобы убрать далекие точки
    # TODO: здесь, вероятно, нужно смотреть не более некоторого количества точек вдоль направления,
    #  а не ограничивать расстояние.
    # radius_upper = 1000.0

    # Стартовые направления
    start_lineZ = np.arange(0, 22, 1.0)

    # !Здесь происходит масштабирование!
    start_lineX = start_x - np.tan(start_angle_x) * start_lineZ
    start_lineY = start_y - np.tan(start_angle_y) * start_lineZ
    start_lineZ = start_lineZ / K

    # print(start_lineZ)

    distX = np.array(
        [_dist_from_line_to_point(start_lineX[0], start_lineZ[0], start_lineX[-1], start_lineZ[-1], x[i], zx[i])
         for i in range(len(x))])
    distY = np.array(
        [_dist_from_line_to_point(start_lineY[0], start_lineZ[0], start_lineY[-1], start_lineZ[-1], y[i], zy[i])
         for i in range(len(y))])

    # - - - - - - - - - - - #
    # Смотрим максимумы энерговыделения вдоль трека
    # Если в каком-то получаем картинку с ветками, выводим её.
    # Проверяем только несколько пиков
    # - - - - - - - - - - - #

    if search_max:
        # Здесь мы берём нефильтрованное событие
        # Выбираем три максимальные позиции
        peaks_sorted = get_max_dedx_track_positions(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y,
                                                     num=num_max_positions, radius=track_radius)[0]
    else:
        # Выбираем только ненулевые значения
        dedx_track_x, dedx_track_y = \
            get_track_dedx(img_x, img_y, start_x, start_y, start_angle_x, start_angle_y, radius=track_radius)
        peaks_sorted = [p for p in range(22) if dedx_track_x[p] + dedx_track_y[p] > 0]
        # print(peaks_sorted)

    vprint("Peaks positions: ", peaks_sorted)

    # Значение, определяющее выбор пика
    # Сейчас это сумма значений функции распределения в проекциях
    if len(peaks_sorted) > 0:
        peaks_rate = np.zeros(len(peaks_sorted))

        interaction_points = dict([(i, []) for i in range(len(peaks_sorted))])
        polar_angles_x = dict([(i, []) for i in range(len(peaks_sorted))])
        polar_angles_y = dict([(i, []) for i in range(len(peaks_sorted))])
    else:
        peaks_rate = [0]

    # Проходим пики в порядке убывания.
    for pk in range(len(peaks_sorted)):
        peak_z_idx = peaks_sorted[pk]

        vprint('Peak position: {0}'.format(peak_z_idx))

        new_x, new_y, new_z = start_lineX[peak_z_idx], start_lineY[peak_z_idx], start_lineZ[peak_z_idx]

        # - - - - - - - - - - - #
        # Будем искать точку взаимодействия.
        # Проходим вдоль прямой (трека первичной частицы) и смотрим, есть ли выходящие из этой точки прямые.
        # При вычислении не учитываем точки исходной прямой: создадим временный массив, в котором они будут удалены.
        # NB: я пробовал не убирать точки с продолжения трека,
        # но тогда в случае неверной идентификации он вносит слишком большой вклад.
        # - - - - - - - - - - - #
        tmp_x, tmp_y = np.copy(x), np.copy(y)
        tmp_zx, tmp_zy = np.copy(zx), np.copy(zy)

        filter_idx_x = distX > radius_lower  # | (zx >= new_z)
        filter_idx_y = distY > radius_lower  # | (zy >= new_z)

        tmp_x = tmp_x[filter_idx_x]
        tmp_y = tmp_y[filter_idx_y]
        tmp_zx = tmp_zx[filter_idx_x]
        tmp_zy = tmp_zy[filter_idx_y]

        (polar_rhoX, polar_phiX) = cartesian2polar(tmp_x - new_x, tmp_zx - new_z)
        (polar_rhoY, polar_phiY) = cartesian2polar(tmp_y - new_y, tmp_zy - new_z)

        if method == 'default':
            image_polarX = _gaussian_sum_1d_weighted_circular(polar_angles_list, polar_phiX, default_sigma,
                                                     np.power(polar_rhoX + weight_shift, weight_power))
            image_polarY = _gaussian_sum_1d_weighted_circular(polar_angles_list, polar_phiY, default_sigma,
                                                     np.power(polar_rhoY + weight_shift, weight_power))
        elif method == 'kde':
            if len(polar_phiX) > 1:
                image_polarX = len(polar_phiX) * scipy.stats.gaussian_kde(
                    polar_phiX, bw_method=default_sigma, weights=np.power(polar_rhoX + weight_shift, weight_power)
                ).evaluate(polar_angles_list)
            else:
                image_polarX = polar_angles_list * 0
            if len(polar_phiY) > 1:
                image_polarY = len(polar_phiY) * scipy.stats.gaussian_kde(
                    polar_phiY, bw_method=default_sigma, weights=np.power(polar_rhoY + weight_shift, weight_power)
                ).evaluate(polar_angles_list)
            else:
                image_polarY = polar_angles_list * 0

        # Ищем пики распределения
        polar_peaksX = scipy.signal.find_peaks(image_polarX, distance=peak_angle_difference)[0][:max_peaks]
        polar_peaksY = scipy.signal.find_peaks(image_polarY, distance=peak_angle_difference)[0][:max_peaks]
        polar_peaksX_values, polar_peaksY_values = image_polarX[polar_peaksX], image_polarY[polar_peaksY]

        vprint(polar_peaksX, polar_peaksY)

        if plot:
            ticks = np.linspace(-np.pi, np.pi, 5, endpoint=True)
            labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']

            fig, ax = plt.subplots(2, 2)
            ax[0][0].scatter(polar_phiX, polar_rhoX, s=5, c='k')
            ax[0][0].set_xlabel(r'$\varphi$, rad.')
            ax[0][0].set_ylabel(r'$\rho$')
            ax[0][0].set_xlim(-np.pi - 0.2, np.pi + 0.2)
            ax[0][0].set_xticks(ticks)
            ax[0][0].set_xticklabels(labels)
            ax[0][0].plot([polar_angles_list[polar_peaksX[i]] for i in range(len(polar_peaksX))], polar_peaksX * 0, 'xr')

            ax[0][1].plot(polar_angles_list, image_polarX, linestyle='-', color='k')
            ax[0][1].plot([polar_angles_list[0], polar_angles_list[-1]], [peak_threshold, peak_threshold], '--r', lw=0.5)
            ax[0][1].set_xlabel(r'$\varphi$, rad.')
            ax[0][1].set_ylabel('Density')
            ax[0][1].set_xticks(ticks)
            ax[0][1].set_xticklabels(labels)
            for i in range(len(polar_peaksX)):
                ax[0][1].plot([polar_angles_list[polar_peaksX[i]], polar_angles_list[polar_peaksX[i]]],
                              [0, polar_peaksX_values[i]], '--r')

            ax[1][0].scatter(polar_phiY, polar_rhoY, s=5, c='k')
            ax[1][0].set_xlabel(r'$\varphi$, rad.')
            ax[1][0].set_ylabel(r'$\rho$')
            ax[1][0].set_xticks(ticks)
            ax[1][0].set_xticklabels(labels)
            ax[1][0].plot([polar_angles_list[polar_peaksY[i]] for i in range(len(polar_peaksY))], polar_peaksY * 0, 'xr')

            ax[1][1].plot(polar_angles_list, image_polarY, linestyle='-', color='k')
            ax[1][1].plot([polar_angles_list[0], polar_angles_list[-1]], [peak_threshold, peak_threshold], '--r', lw=0.5)
            ax[1][1].set_xlabel(r'$\varphi$, rad.')
            ax[1][1].set_ylabel('Density')
            ax[1][1].set_xticks(ticks)
            ax[1][1].set_xticklabels(labels)
            for i in range(len(polar_peaksY)):
                ax[1][1].plot([polar_angles_list[polar_peaksY[i]], polar_angles_list[polar_peaksY[i]]],
                              [0, polar_peaksY_values[i]], '--r')

        polar_peaksX = polar_peaksX[polar_peaksX_values >= peak_threshold]
        polar_peaksY = polar_peaksY[polar_peaksY_values >= peak_threshold]
        polar_peaksX_values = polar_peaksX_values[polar_peaksX_values >= peak_threshold]
        polar_peaksY_values = polar_peaksY_values[polar_peaksY_values >= peak_threshold]

        vprint("Peak number: ", len(polar_peaksX_values), len(polar_peaksY_values))

        # Проверяем, есть ли взаимодействие
        peak_submitted = (len(polar_peaksX_values) > 1) or (len(polar_peaksY_values) > 1)
        # peaks_rate[pk] = len(polar_peaksX_values) + len(polar_peaksY_values) if peak_submitted else 0
        peaks_rate[pk] = sum(polar_peaksX_values) + sum(polar_peaksY_values) if peak_submitted else 0

        if peak_submitted and plot:
            fig, ax = plt.subplots(2, 1)

            ax[0].scatter(x, zx, s=4, c='k')
            ax[0].scatter(tmp_x, tmp_zx, s=2, c='r')
            for angle in polar_angles_list[polar_peaksX]:
                plot_line_by_start_point_and_angle(new_x, new_z, angle, ax=ax[0], color='r')
            ax[0].plot(start_lineX[start_lineZ <= new_z], start_lineZ[start_lineZ <= new_z], color='orange',
                       linestyle='--')

            ax[1].scatter(y, zy, s=4, c='k')
            ax[1].scatter(tmp_y, tmp_zy, s=2, c='r')
            for angle in polar_angles_list[polar_peaksY]:
                plot_line_by_start_point_and_angle(new_y, new_z, angle, ax=ax[1], color='r')
            ax[1].plot(start_lineY[start_lineZ <= new_z], start_lineZ[start_lineZ <= new_z], color='orange',
                       linestyle='--')

            fig.suptitle('Peak rate = {0}'.format(peaks_rate[pk]))
            ax[0].set_title('X')
            ax[1].set_title('Y')

            ax[0].set_xlim((0, 96))
            ax[1].set_xlim((0, 96))
            ax[0].set_ylim((22 / K, 0))
            ax[1].set_ylim((22 / K, 0))

        interaction_points[pk] = (new_x, new_y, new_z)
        polar_angles_x[pk] = polar_peaksX
        polar_angles_y[pk] = polar_peaksY

    if max(peaks_rate) == 0:
        # Рисуем трек до последнего энерговыделения вдоль трека
        if plot_result:
            # TODO: придумать что-то лучше этой затычки
            try:
                z_track_max = min(zx[distX < radius_lower].max(), zy[distY < radius_lower].max())
            except:
                z_track_max = 22

            fig, ax = plt.subplots(2, 1)

            ax[0].imshow(img_x > 0, cmap='Greys')
            ax[0].plot(start_lineX[start_lineZ <= z_track_max], start_lineZ[start_lineZ <= z_track_max] * K,
                       color='orange', linestyle='--')

            ax[1].imshow(img_y > 0, cmap='Greys')
            ax[1].plot(start_lineY[start_lineZ <= z_track_max], start_lineZ[start_lineZ <= z_track_max] * K,
                       color='orange', linestyle='--')

            ax[0].set_title('X')
            ax[1].set_title('Y')

            ax[0].set_xlim((0, 96))
            ax[1].set_xlim((0, 96))
            ax[0].set_ylim((22, 0))
            ax[1].set_ylim((22, 0))

            if output_file is not None:
                plt.savefig(output_file)
                plt.close()

        return (-1, -1, -1), [], []
    else:
        # Пересчитываем результат с учетом хвоста трека первичной частицы
        result_peak_idx = np.argmax(peaks_rate)
        peak_z_idx = peaks_sorted[result_peak_idx]

        vprint("Result peak index {0}".format(result_peak_idx))

        new_x, new_y, new_z = interaction_points[result_peak_idx]

        filter_idx_x = (distX > radius_lower) | (zx >= new_z + radius_lower)
        filter_idx_y = (distY > radius_lower) | (zy >= new_z + radius_lower)

        tmp_x = x[filter_idx_x]
        tmp_y = y[filter_idx_y]
        tmp_zx = zx[filter_idx_x]
        tmp_zy = zy[filter_idx_y]

        (polar_rhoX, polar_phiX) = cartesian2polar(tmp_x - new_x, tmp_zx - new_z)
        (polar_rhoY, polar_phiY) = cartesian2polar(tmp_y - new_y, tmp_zy - new_z)
        image_polarX = _gaussian_sum_1d_weighted_circular(polar_angles_list, polar_phiX, default_sigma,
                                                 np.power(polar_rhoX + weight_shift, weight_power))
        image_polarY = _gaussian_sum_1d_weighted_circular(polar_angles_list, polar_phiY, default_sigma,
                                                 np.power(polar_rhoY + weight_shift, weight_power))

        polar_peaksX = scipy.signal.find_peaks(image_polarX, distance=peak_angle_difference)[0]
        polar_peaksY = scipy.signal.find_peaks(image_polarY, distance=peak_angle_difference)[0]
        polar_peaksX_values, polar_peaksY_values = image_polarX[polar_peaksX], image_polarY[polar_peaksY]
        polar_peaksX = polar_peaksX[polar_peaksX_values >= peak_threshold][:max_peaks]
        polar_peaksY = polar_peaksY[polar_peaksY_values >= peak_threshold][:max_peaks]

        if plot_result:
            fig, ax = plt.subplots(2, 1)

            ax[0].imshow(img_x > 0, cmap='Greys')
            # ax[0].scatter(tmp_x, tmp_zx * K, s=4, c='r')
            for angle in polar_angles_list[polar_peaksX]:
                if np.cos(angle) < 0:
                    plot_line_by_start_point_and_angle(new_x, new_z * K, np.pi - image_angle2real(np.pi - angle), ax=ax[0], color='r')
                else:
                    plot_line_by_start_point_and_angle(new_x, new_z * K, image_angle2real(angle), ax=ax[0], color='r')

            ax[0].plot(start_lineX[start_lineZ <= new_z], start_lineZ[start_lineZ <= new_z] * K, color='orange',
                       linestyle='--')

            ax[1].imshow(img_y > 0, cmap='Greys')
            # ax[1].scatter(tmp_y, tmp_zy * K, s=4, c='r')
            for angle in polar_angles_list[polar_peaksY]:
                if np.cos(angle) < 0:
                    plot_line_by_start_point_and_angle(new_y, new_z * K, np.pi - image_angle2real(np.pi - angle), ax=ax[1], color='r')
                else:
                    plot_line_by_start_point_and_angle(new_y, new_z * K, image_angle2real(angle), ax=ax[1], color='r')
            ax[1].plot(start_lineY[start_lineZ <= new_z], start_lineZ[start_lineZ <= new_z] * K, color='orange',
                       linestyle='--')

            ax[0].set_title('X')
            ax[1].set_title('Y')

            ax[0].set_xlim((0, 96))
            ax[1].set_xlim((0, 96))
            ax[0].set_ylim((22, 0))
            ax[1].set_ylim((22, 0))

            if output_file is not None:
                plt.savefig(output_file)
                plt.close()

        return (new_x, new_y, peak_z_idx), \
            polar_angles_list[polar_angles_x[result_peak_idx]], \
            polar_angles_list[polar_angles_y[result_peak_idx]]
