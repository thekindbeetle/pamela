import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle


class Calorimeter:
    """
    Описание класса калориметра.
    Калориметр состоит из 44 плоскостей (по 22 в каждой проекции)
    Каждая плоскость разбита на 96 стрипов
    """
    strip_num = 96
    plane_num = 44
    data = np.zeros((44, 96))

    def __init__(self, arr):
        self.data = arr

    @classmethod
    def from_projections(cls, proj_x, proj_y):
        data = np.zeros((44, 96))
        data[1::2, :] = proj_x
        data[0::2, :] = proj_y
        return Calorimeter(data)

    @staticmethod
    def plane_quality(strips):
        """
        Качество сигнала в плоскости.
        :param strips: Список энерговыделений по стрипам в плоскости
        :return: True - если проходит по критерию, False - если не проходит.
        """
        # Для определённости считаем, что если все нули, то плоскость критерию не удовлетворяет.
        if max(strips) == 0:
            return False
        amax = strips.argsort()[-3:][::-1]  # Находим три максимума
        q = sum(strips)
        # Либо максимум содержит 90% энерговыделения
        if strips[amax[0]] >= 0.9 * q:
            return True
        # Либо два максимума рядом и содержат 90% энерговыделения
        elif (strips[amax[0]] + strips[amax[1]] >= 0.9 * q) and abs(amax[0] - amax[1]) <= 2:
            return True
        # Либо три максимума рядом и содержат 90% энерговыделения
        elif (strips[amax[0]] + strips[amax[1]] + strips[amax[2]] >= 0.9 * q) and max(amax) - min(amax) <= 3:
            return True
        else:
            return False

    @staticmethod
    def energy2charge(qtot):
        """
        Находим заряд частицы по энерговыделению
        Если точно не определить, выводим -1
        Границы нужно уточнить!
        :param qtot: Энерговыделение
        :return: Заряд ядра
        """
        if qtot <= 100:
            return 1  # протон
        elif 200 <= qtot <= 450:
            return 2  # гелий
        elif 500 <= qtot <= 650:
            return 3  # литий
        elif 900 <= qtot <= 1150:
            return 4  # бериллий
        elif 1400 <= qtot <= 1900:
            return 5  # бор
        elif 2100 <= qtot <= 2600:
            return 6  # углерод
        elif 2800 <= qtot <= 3200:
            return 7  # азот
        elif 3650 <= qtot <= 4400:
            return 8  # кислород
        elif 5500 <= qtot <= 6500:
            return 10  # неон
        else:
            return -1  # не удалось определить

    def x_projection(self):
        """
        Проекция X калориметра - только чётные плоскости.
        [при индексации с нуля - только нечётные]
        :return
        """
        return self.data[1::2, :]

    def y_projection(self):
        """
        Проекция Y калориметра - только нечётные плоскости.
        [при индексации с нуля - только чётные]
        :return
        """
        return self.data[::2, :]

    def plot(self, maxvalue=10, plane38=True, colormap='Blues'):
        """
        Вывести изображение матрицы калориметра
        :param maxvalue: значение, соответствующее максимуму цветовой карты
        :param plane38: закрасить 38-ю плоскость на изображении
        :param colormap: цветовая карта
        """
        fig, axes = plt.subplots(nrows=1, ncols=2)
        for ax in axes.flat:
            ax.grid(True)
            ax.set_xticks(range(0, 96, 1), minor=True)
            ax.set_xticks(range(0, 96, 10), minor=False)
            ax.set_yticks(range(22))

        axes.flat[0].set_yticklabels(range(1, 44, 2))
        axes.flat[1].set_yticklabels(range(2, 45, 2))
        axes.flat[1].imshow(self.x_projection(), aspect=4, interpolation='none', cmap=colormap,
                            norm=colors.Normalize(0, maxvalue))
        if plane38:
            axes.flat[1].add_patch(Rectangle((0, 17.5), 96, 1, facecolor=[1, 0, 0, 0.5]))
        im = axes.flat[0].imshow(self.y_projection(), aspect=4, interpolation='none', cmap=colormap,
                                 norm=colors.Normalize(0, maxvalue))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.set_size_inches((12, 5))
        plt.show()

    def qtotpl(self, plnum):
        """
        Энерговыделение в данной плоскости.
        :param plnum: номер плоскости (1 - 44)
        """
        return sum(self.data[plnum - 1])

    def qtot_range(self, start, end):
        """
        Энерговыделение в диапазоне плоскостей
        :param start: номер начальной плоскости
        :param end: номер конечной плоскости (включительно)
        """
        s = 0
        for plnum in range(start, end + 1):
            if plnum != 38:
                s += self.qtotpl(plnum)
        return s

    def qtot(self):
        """
        Полное энерговыделение в калориметре
        """
        return np.sum(self.data)

    def qtrack_x(self, track_x, r=2):
        """
        Список энерговыделений вдоль трека (проекция X).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке.
        :param r: радиус, в котором смотрим энерговыделение.
        """
        data = self.x_projection()
        cyl_left = [max(track_x[i] - r, 0) if track_x[i] != -1 else -1 for i in range(22)]
        cyl_right = [min(track_x[i] + r, 95) if track_x[i] != -1 else -1 for i in range(22)]
        result = np.zeros(22)
        for i in range(22):
            result[i] = np.sum(data[i][cyl_left[i]:cyl_right[i] + 1]) if cyl_left[i] != -1 else 0
        return result

    def qtrack_y(self, track_y, r=2):
        """
        Список энерговыделений вдоль трека (проекция Y).
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке.
        :param r: радиус, в котором смотрим энерговыделение.
        """
        data = self.y_projection()
        cyl_left = [max(track_y[i] - r, 0) if track_y[i] != -1 else -1 for i in range(22)]
        cyl_right = [min(track_y[i] + r, 95) if track_y[i] != -1 else -1 for i in range(22)]
        result = np.zeros(22)
        for i in range(22):
            result[i] = np.sum(data[i][cyl_left[i]:cyl_right[i] + 1]) if cyl_left[i] != -1 else 0
        return result

    def qtrack(self, track_x, track_y, r=2):
        """
        Список энерговыделений вдоль трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qx = self.qtrack_x(track_x, r)
        qy = self.qtrack_y(track_y, r)
        result = np.zeros(44)
        for i in range(22):
            result[2 * i] = qy[i]
            result[2 * i + 1] = qx[i]
        return result

    def d_qtrack(self, track_x, track_y, r=2):
        """
        Производная по количеству энерговыделений вдоль трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qtrack = self.qtrack(track_x, track_y, r)
        dq = qtrack[1:] - qtrack[:-1]
        result = np.zeros(44)
        result[1:] = dq
        return result

    def d_qtrack_x(self, track_x, r=2):
        """
        Производная по количеству энерговыделений вдоль трека (проекция X).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qx = self.qtrack_x(track_x, r)
        dq = qx[1:] - qx[:-1]
        result = np.zeros(22)
        result[1:] = dq
        return result

    def d_qtrack_y(self, track_y, r=2):
        """
        Производная по количеству энерговыделений вдоль трека (проекция Y).
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qx = self.qtrack_y(track_y, r)
        dq = qx[1:] - qx[:-1]
        result = np.zeros(22)
        result[1:] = dq
        return result

    def ntrack_x(self, track_x, r=2):
        """
        Количество сработавших стрипов вдоль трека (проекция X).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке.
        :param r: радиус, в котором смотрим стрипы.
        """
        data = self.x_projection()
        cyl_left = [max(track_x[i] - r, 0) if track_x[i] != -1 else -1 for i in range(22)]
        cyl_right = [min(track_x[i] + r, 95) if track_x[i] != -1 else -1 for i in range(22)]
        result = np.zeros(22)
        for i in range(22):
            result[i] = np.count_nonzero(data[i][cyl_left[i]:cyl_right[i] + 1]) if cyl_left[i] != -1 else 0
        return result

    def ntrack_y(self, track_y, r=2):
        """
        Количество сработавших стрипов вдоль трека (проекция Y).
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке.
        :param r: радиус, в котором смотрим стрипы.
        """
        data = self.y_projection()
        cyl_left = [max(track_y[i] - r, 0) if track_y[i] != -1 else -1 for i in range(22)]
        cyl_right = [min(track_y[i] + r, 95) if track_y[i] != -1 else -1 for i in range(22)]
        result = np.zeros(22)
        for i in range(22):
            result[i] = np.count_nonzero(data[i][cyl_left[i]:cyl_right[i] + 1]) if cyl_left[i] != -1 else 0
        return result

    def ntrack(self, track_x, track_y, r=2):
        """
        Количество сработавших стрипов вдоль трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        nx = self.ntrack_x(track_x, r)
        ny = self.ntrack_y(track_y, r)
        result = np.zeros(44)
        for i in range(22):
            result[2 * i] = ny[i]
            result[2 * i + 1] = nx[i]
        return result

    def ntrack_out(self, track_x, track_y, r=2):
        """
        Количество сработавших стрипов вне трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        ntots = (self.data > 0).sum(axis=1)
        ntrack = self.ntrack(track_x, track_y, r)
        return ntots - ntrack

    def ntrack_out_x(self, track_x, r=2):
        """
        Количество сработавших стрипов вне трека (проекция X).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param r: радиус, в котором смотрим стрипы.
        """
        ntotsX = (self.x_projection() > 0).sum(axis=1)
        ntrackX = self.ntrack_x(track_x, r)
        return ntotsX - ntrackX

    def ntrack_out_y(self, track_y, r=2):
        """
        Количество сработавших стрипов вне трека (проекция Y).
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        ntotsY = (self.y_projection() > 0).sum(axis=1)
        ntrackY = self.ntrack_y(track_y, r)
        return ntotsY - ntrackY

    def qtrack_out(self, track_x, track_y, r=2):
        """
        Количество выделенной энергии вне трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        qtots = self.data.sum(axis=1)
        qtrack = self.qtrack(track_x, track_y, r)
        return qtots - qtrack

    def d_ntrack(self, track_x, track_y, r=2):
        """
        Производная по количеству сработавших стрипов вдоль трека (обе проекции сразу).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        ntrack = self.ntrack(track_x, track_y, r)
        dn = ntrack[1:] - ntrack[:-1]
        result = np.zeros(44)
        result[1:] = dn
        return result

    def ratio_qtrack_x(self, track_x, r=2):
        """
        Отношение энерговыделения: последующее к предыдущему (в некотором радиусе).
        Рассматриваем только проекцию X.
        Нулевую плоскость заменяем предыдущей. Отношение 0/0 заменяем на 0.
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qtrack = self.qtrack_x(track_x, r)
        for i in range(21):
            if qtrack[i] == 0:
                qtrack[i] = qtrack[i + 1]
        rq = qtrack[1:] / qtrack[:-1]
        result = np.ones(22)
        result[1:] = rq
        return result

    def ratio_qtrack_y(self, track_y, r=2):
        """
        Отношение энерговыделения: последующее к предыдущему (в некотором радиусе).
        Рассматриваем только проекцию Y.
        Нулевую плоскость заменяем предыдущей. Отношение 0/0 заменяем на 0.
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qtrack = self.qtrack_y(track_y, r)
        for i in range(21):
            if qtrack[i] == 0:
                qtrack[i] = qtrack[i + 1]
        rq = qtrack[1:] / qtrack[:-1]
        result = np.ones(22)
        result[1:] = rq
        return result

    def ratio_qtrack(self, track_x, track_y, r=2):
        """
        Отношение энерговыделения: последующее к предыдущему (в некотором радиусе).
        Нулевую плоскость заменяем предыдущей. Отношение 0/0 заменяем на 0.
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим энерговыделение.
        """
        qtrack = self.qtrack(track_x, track_y, r)
        for i in range(43):
            if qtrack[i] == 0:
                qtrack[i] = qtrack[i + 1]
        rq = qtrack[1:] / qtrack[:-1]
        result = np.ones(44)
        result[1:] = rq
        return result

    def d_ntrack_x(self, track_x, r=2):
        """
        Производная по количеству сработавших стрипов вдоль трека (проекция X).
        Радиус по умолчанию равен 2.
        :param track_x: список номеров стрипов в треке (проекция X).
        :param r: радиус, в котором смотрим стрипы.
        """
        nx = self.ntrack_x(track_x, r)
        dn = nx[1:] - nx[:-1]
        result = np.zeros(22)
        result[1:] = dn
        return result

    def d_ntrack_y(self, track_y, r=2):
        """
        Производная по количеству сработавших стрипов вдоль трека (проекция Y).
        Радиус по умолчанию равен 2.
        :param track_y: список номеров стрипов в треке (проекция Y).
        :param r: радиус, в котором смотрим стрипы.
        """
        nx = self.ntrack_y(track_y, r)
        dn = nx[1:] - nx[:-1]
        result = np.zeros(22)
        result[1:] = dn
        return result

    def qpl_median(self):
        """
        Медиана энерговыделения в плоскости
        """
        return np.median(self.data.sum(axis=1))

    def save(self, file):
        """
        Сохранить данные события в CSV-файл
        """
        np.savetxt(file, self.data, delimiter="\t", fmt='%.2f')

    def strips(self):
        """
        Количество сработавших стрипов.
        """
        return np.count_nonzero(self.data)

    def distance(self):
        """
        Расстояние, которое частица проходит в калориметре (зависит от угла попадания частицы).
        Глубина калориметра: 57.2 мм
        Ширина стрипа 2.4 мм
        возвращаем относительное расстояние, за 1 считаем 57.2 мм (глубина калориметра).

        Зависимость энерговыделения от значения distance слабая.
        """
        dx = np.argmax(self.data[0]) - np.argmax(self.data[42])
        dy = np.argmax(self.data[1]) - np.argmax(self.data[43])
        return np.sqrt((dx ** 2 + dy ** 2) * 2.4 ** 2 + 57.2 ** 2) / 57.2

    def print_qtots(self):
        """
        Вывести энерговыделения по плоскостям
        """
        for pl in range(1, 45):
            print("{0}: {1}".format(pl, self.qtotpl(pl)), end='/')
        print()

    def event_quality(self, err=True):
        # Проверяем, для скольких плоскостей критерий выполнен
        sel = [Calorimeter.plane_quality(self.data[idx]) for idx in range(44)]
        penalty = 0
        if err:
            # В последних плоскостях должен быть хороший сигнал
            err = [-10 * int(not Calorimeter.plane_quality(self.data[idx])) for idx in range(38, 44)]
            penalty = sum(err)
        return sum(sel) + penalty

    def identify_charge_if_not_interacting(self, threshold=36, err=True, stability=True, ratio=10):
        """
        Определить заряд частицы, если она проходит калориметр без взаимодействия.
        Иначе — вернуть -1
        threshold — минимальное количество плоскостей с хорошим сигналом
        err — необходимость наличия сигнала в 39-44 плоскостях
        stability — сигнал в последних и первых плоскостях не должен сильно отличаться
        ratio — максимальное допустимое отношение сигналов в сработавших плоскостях
        """
        # Количество плоскостей с хорошим сигналом не меньше порога
        if self.event_quality(err=err) < threshold:
            return -1

        qpls = self.data.sum(axis=1)  # Отношение макс. / мин. сигнал не должно быть большим
        minval = np.min(qpls[np.nonzero(qpls)])
        maxval = np.max(qpls)

        if maxval > minval * ratio:
            return -1

        if stability:
            first_q = self.qtot_range(1, 5)
            last_q = self.qtot_range(40, 44)

            if first_q > last_q * 1.25:
                return -1

            if last_q > first_q * 1.25:
                return -1

        charge = Calorimeter.energy2charge(sum(qpls))
        if charge == 1:
            if self.filter_proton():
                return 1
            else:
                return -1
        else:
            return charge

    def filter_proton(self):
        """
        Проверяем, может ли событие быть протоном.
        """
        # фильтруем протоны по критерию Андрея
        event_mask = (self.data > 0)
        stripcounts = np.sum(event_mask, axis=1)
        if np.sum(stripcounts) > 49 or max(stripcounts) > 3 or sum(stripcounts == 2) > 4 or sum(stripcounts == 3) > 1:
            return False
        for i in range(44):
            k = np.where(event_mask[i, :] is True)[0]
            # к пустым бинам нет претензий
            if len(k) == 0:
                continue
            # между стрипами большое расстояние
            elif np.max(k) - np.min(k) > 2:
                return False
        return True
