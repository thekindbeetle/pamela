import numpy as np
import scipy.optimize as opt
from skimage.draw import line_nd
from skimage.morphology import binary_dilation, binary_erosion
from scipy.spatial import distance_matrix
import ot
import time

DILATION_FOOTPRINT = np.ones((3, 4))  # Жирная окрестность
DILATION_FOOTPRINT3D = np.ones((2, 2, 1))  # Жирная окрестность в 3D


def x_proj_binary(C):
    return C.any(axis=1)


def y_proj_binary(C):
    return C.any(axis=0)


def get_exit_point(start_x, start_y, start_z, dir_x, dir_y, dir_z):
    """
    Находит точку вылета частицы из калориметра (если она не останавливается)
    Параметры:
        Начало луча: start_x, start_y, start_z.
        Направление луча: dir_x, dir_y, dir_z.
    Возвращает:
        (last_x, lasty, last_z)
    """
    ray_dir = np.array([dir_x, dir_y, dir_z], dtype=float)
    ray_origin = np.array([start_x, start_y, start_z], dtype=float)
    box_min = np.array([0, 0, 0])
    box_max = np.array([95, 95, 21])

    # Избегаем деления на 0
    # ray_dir[ray_dir == 0] = 1e-10

    t_exit = np.min(np.maximum((box_min - ray_origin) / ray_dir, (box_max - ray_origin) / ray_dir))
    point_exit = ray_origin + t_exit * ray_dir

    return point_exit


def get_cells_without_run_aspect(start_x, start_y, start_z, theta, phi):
    """
    Список ячеек калориметра, пересекаемых лучом.
    Длина луча не устанавливается.
    Учитывается разнице в масштабе по осям X / Z
    @param start_x: Координата X стартовой точки
    @param start_y: Координата Y стартовой точки
    @param start_z: Координата Z стартовой точки
    @param theta: Зенитный угол, диапазон 0..pi
    @param phi: Азимутальный угол, диапазон -pi..pi
    @return:
    """
    if theta < np.pi / 2:
        end = get_exit_point(start_x, start_y, start_z, np.tan(theta) * np.cos(phi), np.tan(theta) * np.sin(phi), 1)
    else:
        end = get_exit_point(start_x, start_y, start_z, -np.tan(theta) * np.cos(phi), -np.tan(theta) * np.sin(phi), -1)

    # Растягиваем калориметр в три раза
    end[2] = end[2] * 3
    start = start_x, start_y, start_z * 3

    ii, jj, kk = line_nd(start, end, endpoint=True)
    idx = np.mod(kk, 3) == 0  # Нас интересуют только плоскости с номером, делящимся на три.
    ii = ii[idx]
    jj = jj[idx]
    kk = kk[idx] // 3

    # Объединяем в список кортежей
    line = [ii, jj, kk]

    return line


def get_cells_with_run_aspect(start_x, start_y, start_z, theta, phi, run):
    """
    Список ячеек калориметра, пересекаемых лучом.
    Устанавливается максимальная длина луча.
    Учитывается разнице в масштабе по осям X / Z
    @param start_x: Координата X стартовой точки
    @param start_y: Координата Y стартовой точки
    @param start_z: Координата Z стартовой точки
    @param theta: Зенитный угол, диапазон 0..pi
    @param phi: Азимутальный угол, диапазон -pi..pi
    @param run: Максимальная длина луча
    @return:
    """
    run = int(run)
    line = get_cells_without_run_aspect(start_x, start_y, start_z, theta, phi)
    if run < len(line[0]):
        line = [line[0][:run], line[1][:run], line[2][:run]]
    return line


def _generate_event_without_runs(N, start_x, start_y, start_z, theta, phi):
    """
    Создание события пролёта частицы через трёхмерный калориметр.
    @param N: Количество частиц (включая первичную)
    @param start_x: Точка взаимодействия (координата X).
    @param start_y: Точка взаимодействия (координата Y).
    @param start_z: Точка взаимодействия (координата Z).
    @param theta: Список сферических углов разлёта (зенитный угол от 0 до pi)
    @param phi: Список сферических углов разлёта (азимутальный угол от -pi до pi)
    @return:
    """
    C = np.zeros((96, 96, 22), dtype=int)

    lines = dict()

    for line_num in range(N):
        line = get_cells_without_run_aspect(start_x, start_y, start_z, theta[line_num], phi[line_num])
        C[line[0], line[1], line[2]] = 1

    return C


def _generate_N_event_without_runs(params, N):
    return _generate_event_without_runs(N, *params[0:3], params[3: 3 + N], params[3 + N: 3 + 2 * N])


# Стартовая точка -- это точка взаимодействия, из неё разбегаются частицы
# StartX 0..95, StartY 0..95, StartZ 0..21, theta 0..pi, phi -pi..pi, run 2..21
def _generate_event_with_runs(N, start_x, start_y, start_z, theta, phi, run):
    """
    Создание события пролёта частицы через трёхмерный калориметр.
    @param N: Количество частиц (включая первичную)
    @param start_x: Точка взаимодействия (координата X).
    @param start_y: Точка взаимодействия (координата Y).
    @param start_z: Точка взаимодействия (координата Z).
    @param theta: Список сферических углов разлёта (зенитный угол от 0 до pi)
    @param phi: Список сферических углов разлёта (азимутальный угол от -pi до pi)
    @param run: Список пробегов частиц
    @return:
    """
    C = np.zeros((96, 96, 22), dtype=int)

    for line_num in range(N):
        line = get_cells_with_run_aspect(start_x, start_y, start_z, theta[line_num], phi[line_num], run[line_num])
        C[line[0], line[1], line[2]] = 1

    return C


# Создаём событие для N порождённых частиц по списку параметров
# Универсальная
def _generate_N_event_with_runs(params, N):
    return _generate_event_with_runs(N, *params[0:3], params[3: 3 + N], params[3 + N: 3 + 2 * N],
                                     params[3 + 2 * N: 3 + 3 * N])


#
def wasserstein_distance(mat1, mat2):
    """
    Транспортная метрика (Вассерштейна) - расстояние между матрицами.
    Она энергозатратная, но для разреженных матриц вроде бы не критично
    @param mat1:
    @param mat2:
    @return:
    """
    # Get coordinates of 1s in each matrix
    coords1 = np.argwhere(mat1 == 1)  # shape (N, 3)
    coords2 = np.argwhere(mat2 == 1)  # shape (M, 3)

    if len(coords1) == 0 or len(coords2) == 0:
        distance = np.inf
    else:
        # Compute Euclidean cost matrix (distance between all pairs)
        cost_matrix = distance_matrix(coords1, coords2)

        # Uniform weights (since all 1s are equally important)
        weights1 = np.ones(len(coords1)) / len(coords1)
        weights2 = np.ones(len(coords2)) / len(coords2)

        # Compute Wasserstein distance
        distance = ot.emd2(weights1, weights2, cost_matrix)
    return distance


def cover_distance(mat_init, mat_cover):
    """
    Расстояние покрытия (сколько целевых стрипов не покрыто).
    Несимметричное.
    @param mat_init: Матрица, которую покрываем
    @param mat_cover: Матрица, которой покрываем
    @return:
    """
    return np.sum(mat_init > mat_cover)


# Расстояние покрытия (сколько целевых стрипов не покрыто).
# Применяем последовательно несколько раз эрозию, для каждой итерации считаем метрику покрытия, затем суммируем
def cover_erosion_distance(mat_init, mat_cover, num_iterations=5):
    result = np.sum(mat_init > mat_cover)
    mat_init_eroded = mat_init
    for i in range(num_iterations):
        mat_init_eroded = binary_erosion(mat_init_eroded)
        result += np.sum(mat_init_eroded > mat_cover)
    return result


# С дилатацией и фиксированными пробегами
def _objective_N_without_runs(params, to_x, to_y, N, distance=wasserstein_distance):
    N, startx, starty, startz, theta_part, phi_part = N, params[0], params[1], params[2], params[3: 3 + N], params[
                                                                                                            3 + N: 3 + 2 * N]
    E = _generate_event_without_runs(N, startx, starty, startz, theta_part, phi_part)
    return distance(to_x, binary_dilation(x_proj_binary(E), footprint=DILATION_FOOTPRINT)) + \
           distance(to_y, binary_dilation(y_proj_binary(E), footprint=DILATION_FOOTPRINT))


def reconstruct_track(Xproj, Yproj, particle_num=5, threshold=0.1, verbose=True, distance_func=cover_distance):
    """
    @param Xproj: проекция XZ
    @param Yproj: проекция YZ
    @param particle_num: количество частиц в модели (включая первичную)
    @param threshold: доля стрипов, которая может быть не покрыта.
    @param verbose:
    @param distance_func: Используемая метрика расстояния
    @return:
    """
    assert particle_num >= 2, "Частиц должно быть минимум две."

    threshold_value = (Xproj.sum() + Yproj.sum()) * threshold
    max_z = max(np.nonzero(Xproj.sum(axis=0))[0][-1],
                np.nonzero(Yproj.sum(axis=0))[0][-1])  # Последняя плоскость с энерговыделением
    z_bound = int(max_z * 2 / 3)

    if verbose:
        print(f"Maximum Z value = {max_z}; Z bound = {z_bound}")
        print(f"Threshold value: {int(threshold_value)} cells")

    # Возможно, нужно выбрать стартовую популяцию явно
    start_time = time.time()  # Засекаем начальное время

    def diff_callback(xk, convergence):
        current_min = _objective_N_without_runs(xk, Xproj, Yproj, particle_num, distance=distance_func)
        print(r"{0:.0f} / ".format(current_min), end="")
        return current_min <= threshold_value  # Stop if condition met

    # ! Я поправил bounds для Zint
    result_params = opt.differential_evolution(
        _objective_N_without_runs, args=(Xproj, Yproj, particle_num, distance_func),
        # init = 'random',
        # init = 'sobol',
        init='latinhypercube',
        bounds=[(0, 95), (0, 95), (1, z_bound), (2 * np.pi / 3, np.pi), *[(0, np.pi)] * (particle_num - 1),
                *[(-np.pi, np.pi)] * particle_num],
        callback=diff_callback,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.9,
        # recombination=0.7,
        maxiter=1000,
        popsize=20,
        tol=1e-3
    )

    end_time = time.time()  # Засекаем начальное время
    elapsed_time = end_time - start_time  # Вычисляем разницу

    print()
    print(f"Время выполнения: {elapsed_time:.4f} секунд")

    print("Wasserstein distance = {0:.3f}".format(
        _objective_N_without_runs(result_params.x, Xproj, Yproj, particle_num, distance=wasserstein_distance)))
    print("Cover distance = {0:.3f}".format(
        _objective_N_without_runs(result_params.x, Xproj, Yproj, particle_num, distance=cover_distance)))
    print("-----------")

    return result_params
