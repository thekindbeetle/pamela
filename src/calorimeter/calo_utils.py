import numpy as np
import matplotlib.pyplot as plt
from src.util.coord_transform import polar2cartesian


# Коэффициент горизонтального сжатия при подсчёте расстояний.
# Расстояние между плоскостями в K раз больше расстояния между стрипами
# Здесь мы считаем, что все эти расстояния равны (что недалеко от истины).
x_scale = 0.244  # расстояние между стрипами (см)
z_scale = 0.909  # расстояние между серединами плоскостей (см)
K = x_scale / z_scale

# Максимальный угол влёта в калориметр 0.4 радиан (установлено опытным путём)
max_real_angle = 0.4


def image_angle2real(theta):
    """
    Перевод угла на изображении в реальный угол (масштаб по оси Z)
    :param theta:
    :return:
    """
    return np.arctan(K * np.tan(theta))


def real_angle2image(theta):
    """
    Перевод реального угла в угол на изображении (масштаб по оси Z)
    :param theta:
    :return:
    """
    return np.arctan(1 / K * np.tan(theta))


# Максимальный угол влёта на картинке (отмасштабированный)
max_image_angle = real_angle2image(max_real_angle)


def get_angle_by_projections(theta_x, theta_y):
    """
    Восстанавливаем угол по углам в проекциях
    :param theta_x: Угол влёта в проекции X
    :param theta_y: Угол влёта в проекции Y
    :return: Зенитный угол влёта
    """
    return np.arctan(np.sqrt(np.power(np.tan(theta_x), 2) + np.power(np.tan(theta_y), 2)))


def get_projection_angles(theta, phi):
    """
    Считаем углы в проекциях по углам влёта
    :param theta: Зенитный угол влёта
    :param phi: Азимутальный угол влёта
    :return: Углы влёта в проекциях X, Y
    """
    return -np.arctan(np.tan(theta) * np.cos(phi)), -np.arctan(np.tan(theta) * np.sin(phi))


def plot_line_by_start_point_and_angle(x, z, angle, ax=None, color='r'):
    """
    Рисуем линию в калориметре по стартовой точке и углу
    :param x: Стартовая точка (координата X или Y)
    :param z: Стартовая точка (координата Z)
    :param angle: Угол влёта в проекцию
    :param ax: Экземпляр осей, в которых рисуем картинку (если None, то создаём новые)
    :param color: Цвет линии.
    :return:
    """
    if ax is None:
        ax = plt.gca()

    new_x, new_z = polar2cartesian(100.0, angle)
    ax.plot([x, x + new_x], [z, z + new_z], color=color)
