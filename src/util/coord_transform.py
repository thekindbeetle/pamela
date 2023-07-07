import numpy as np


def polar2cartesian(rho, phi):
    """
    Переход к декартовым координатам из полярных
    :param rho: полярное расстояние
    :param phi: полярный угол
    :return: Декартовы координаты (X, Y)
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cartesian2polar(x, y):
    """
    Переход к полярным координатам
    :param x: декартова координата X
    :param y: декартова координата Y
    :return: Полярные координаты (rho, phi)
    """
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return rho, phi
