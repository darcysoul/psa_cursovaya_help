import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f, chi2, t
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabulate import tabulate


def load_data(data_paths):
    """Загружает данные из заданных путей."""
    data = {column: np.loadtxt(path, converters={0: lambda x: float(x.replace(',', '.'))}) for column, path in data_paths.items()}
    df = pd.DataFrame(data)
    df['timestamp'] = pd.period_range(start='2019-01', end='2024-12', freq='M')
    df.set_index('timestamp', inplace=True)
    return df


def compute_discrete_distribution(data):
    """Вычисляет дискретное распределение."""
    distribution = {}
    for value in data:
        distribution[value] = distribution.get(value, 0) + 1
    return dict(sorted(distribution.items()))


def compute_interval_distribution(data):
    """Вычисляет интервальное распределение."""
    discrete_distribution = compute_discrete_distribution(data)
    num_intervals = int(np.ceil(1 + 3.222 * np.log10(len(data))))
    interval_length = int(np.ceil((max(data) - min(data)) / num_intervals))

    intervals = [(i, i + interval_length) for i in range(min(data), max(data), interval_length)]
    frequencies = [0] * len(intervals)

    for i, interval in enumerate(intervals):
        for value, count in discrete_distribution.items():
            if interval[0] <= value < interval[1]:
                frequencies[i] += count

    return intervals, frequencies, num_intervals, interval_length


def plot_histogram(data, num_bins):
    """Строит гистограмму данных."""
    plt.hist(data, bins=num_bins, range=(min(data), max(data)), edgecolor='black')
    plt.xlabel('Рублей')
    plt.ylabel('Частота')
    plt.title('Гистограмма потребительских расходов населения')
    plt.show()


def compute_normal_distribution_properties(data):
    """Вычисляет свойства нормального распределения. Проверяет нормальность распределения."""
    mean_value = np.mean(data)
    std_dev = np.std(data)

    sigma_68 = np.mean((mean_value - std_dev <= data) & (data <= mean_value + std_dev)) * 100
    sigma_95 = np.mean((mean_value - 2 * std_dev <= data) & (data <= mean_value + 2 * std_dev)) * 100
    sigma_99 = np.mean((mean_value - 3 * std_dev <= data) & (data <= mean_value + 3 * std_dev)) * 100

    if sigma_68 > 68 and sigma_95 > 95 and sigma_99 > 99.7:
        is_norm_distribution = 'Распределение нормальное по правилу 3-х сигм'
    else:
        is_norm_distribution = 'Распределение не нормальное по правилу 3-х сигм'

    return sigma_68, sigma_95, sigma_99, is_norm_distribution


def compute_normal_distribution_pearson(data, intervals, frequencies, interval_length):
    """Проверяет нормальность распределения по критерию Пирсона."""
    n = sum(frequencies)
    midpoints = [(left + right) / 2 for left, right in intervals]

    def mean_midpoints(intervals, frequencies):
        return np.sum([((left + right) / 2) * frequencies[i] for i, (left, right) in enumerate(intervals)]) / sum(frequencies)

    def variance_midpoints(intervals, frequencies):
        mean_value = mean_midpoints(intervals, frequencies)
        return np.sum([(((left + right) / 2) - mean_value) ** 2 * frequencies[i] for i, (left, right) in enumerate(intervals)]) / sum(frequencies)

    def std_dev_midpoints(intervals, frequencies):
        return np.sqrt(variance_midpoints(intervals, frequencies))

    standardized_midpoints = [(point - mean_midpoints(intervals, frequencies)) / std_dev_midpoints(intervals, frequencies) for point in midpoints]

    def laplace_function(x):
        return np.exp(- (x ** 2 / 2)) / (np.sqrt(2 * np.pi))

    theoretical_frequencies = [(interval_length * n / std_dev_midpoints(intervals, frequencies)) * laplace_function(ui) for ui in standardized_midpoints]

    chi_squared_observed = np.sum([(frequencies[i] - j)**2 / j for i, j in enumerate(theoretical_frequencies)])

    #  Количество оцениваемых параметров для нормального распределения описывается двумя параметрами (среднее и стандартное отклонение)
    r = 2
    degrees_of_freedom = len(intervals) - r - 1
    alpha = 0.05
    chi_squared_critical = chi2.ppf(1 - alpha, degrees_of_freedom)

    if chi_squared_observed < chi_squared_critical:
        is_chi_norm_distribution = 'Нет оснований отвергать гипотезу H0 по критерию Пирсона'
    else:
        is_chi_norm_distribution = 'Распределение не является нормальным по критерию Пирсона'

    return chi_squared_observed, chi_squared_critical,  chi_squared_observed / chi_squared_critical, is_chi_norm_distribution


def analyze_correlation(df):
    """Визуализирует матрицу корреляций."""
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", annot_kws={"size": 10})
    plt.show()
    print(f'Детерминант матрицы парных корреляций: {np.linalg.det(df.corr().to_numpy())}')


def fisher_test(y_true, X, model, num_samples, num_features):
    """Выполняет тест Фишера."""
    S2_fact = np.sum((model.predict(X) - np.mean(y_true)) ** 2) / num_features
    S2_e = np.sum((y_true - model.predict(X)) ** 2) / (num_samples - num_features - 1)
    F_statistic = S2_fact / S2_e
    alpha = 0.05
    critical_value = f.ppf(1 - alpha, num_features, num_samples - num_features - 1)
    p_value = 1 - f.cdf(F_statistic, num_features, num_samples - num_features - 1)
    return F_statistic, critical_value, p_value


def student_test(model, num_samples, num_features):
    """Определяет критическое значение критерия Стьюдента."""
    k = num_samples - num_features - 1
    alpha = 0.05
    critical_value = t.ppf(1 - alpha / 2, k)
    return critical_value


def evaluate_model(y_true, y_predicted, num_samples, num_features, model):
    """Оценивает качество модели."""
    mse = mean_squared_error(y_true, y_predicted)
    mae = mean_absolute_error(y_true, y_predicted)
    r_squared = model.rsquared
    r_squared_adj = 1 - (num_samples - 1) / (num_samples - num_features - 1) * (1 - r_squared)

    evaluation_metrics = [
        ["Среднеквадратичная ошибка (MSE)", f"{mse:.4f}"],
        ["Средняя абсолютная ошибка (MAE)", f"{mae:.4f}"],
        ["Коэффициент детерминации", f"{r_squared:.3f}"],
        ["Адаптивный коэффициент детерминации", f"{r_squared_adj:.3f}"]
    ]
    print(tabulate(evaluation_metrics, headers=["Метрика", "Значение"], tablefmt="fancy_grid"))


def logic_2():
    # Загрузка данных
    data_paths = {
        "x1": './src/punkt2/x1.txt',
        # "x2": './src/punkt2/x2.txt',
        "x3": './src/punkt2/x3.txt',
        # "x4": './src/punkt2/x4.txt',
        "x5": './src/punkt2/x5.txt',
        "y": './src/punkt2/y.txt'
        }
    df = load_data(data_paths)
    target_values = df['y'].values.astype(int)

    # Интервальное распределение
    print('Интервальное распределение')
    intervals, frequencies, num_intervals, interval_length = compute_interval_distribution(target_values)
    interval_table = [["Интервал", "Частота"]]
    for i, interval in enumerate(intervals):
        interval_table.append([f"[{interval[0]}, {interval[1]})", frequencies[i]])
    print(tabulate(interval_table, headers="firstrow", tablefmt="fancy_grid"), end='\n\n')

    # Построение гистограммы
    plot_histogram(target_values, num_intervals)

    # Нормальное распределение по теореме 3-х сигм
    print('Нормальное распределение по теореме 3-х сигм')
    sigma_68, sigma_95, sigma_99, is_norm_distribution = compute_normal_distribution_properties(target_values)

    sigma_table = [
        ["Интервал", "Процент"],
        ["1 сигма", f"{sigma_68:.2f}"],
        ["2 сигмы", f"{sigma_95:.2f}"],
        ["3 сигмы", f"{sigma_99:.2f}"]
    ]
    print(tabulate(sigma_table, headers="firstrow", tablefmt="fancy_grid"))
    print(is_norm_distribution, end='\n\n')

    # Нормальное распределение по критерию Пирсона
    print('Нормальное распределение по критерию Пирсона')
    chi_squared_observed, chi_squared_critical, chi_squared_ratio, is_chi_norm_distribution = compute_normal_distribution_pearson(target_values, intervals, frequencies, interval_length)

    pearson_table = [
        ["Показатель", "Значение"],
        ["Хи-квадрат наблюдаемое", f"{chi_squared_observed:.3f}"],
        ["Хи-квадрат критическое", f"{chi_squared_critical:.3f}"],
        ["Рассчитанное значение", f"{chi_squared_ratio:.3f}"]
    ]
    print(tabulate(pearson_table, headers="firstrow", tablefmt="fancy_grid"), end='\n')
    print(is_chi_norm_distribution, end='\n\n')

    # Анализ корреляций
    analyze_correlation(df)

    # Составляем матрицу признаков и вектор ответов
    X = df.drop('y', axis=1)
    y = df['y']

    # Разделение данных на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

    # Добавление константы к матрице признаков для МНК
    X_train_mn = sm.add_constant(X_train)
    X_test_mn = sm.add_constant(X_test)

    # Реализация МНК для тренировочных данных
    model = sm.OLS(y_train, X_train_mn).fit()

    # Получение прогнозов для тестовых данных
    y_predicted = model.predict(X_test_mn)

    # Выполнение теста Фишера
    F_statistic, critical_value, p_value = fisher_test(y_test, X_test_mn, model, len(y_test), X_test_mn.shape[1] - 1)

    print(f"F-критерий: {F_statistic:.3f}")
    print(f"Критическое значение: {critical_value:.3f}")
    print(f"P-значение: {p_value:.3f}")
    print()

    # Вывод результатов регрессии
    result_summary = model.summary()
    coefficients_table = pd.DataFrame(result_summary.tables[1].data[1:], columns=result_summary.tables[1].data[0])
    coefficients_table.columns = [' ', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']

    # Выполнение теста Стьюдента
    print('Анализ статистической значимости коэффициентов уравнения регрессии:')
    print(tabulate(coefficients_table, headers='keys', tablefmt='fancy_grid'))

    critical_value_s = student_test(model, len(y_test), X_test_mn.shape[1] - 1)
    print("Критическое значение T-критерия:", critical_value_s, "\n")

    coefficients_table.to_csv('./src/punkt2/coefficients_table.csv', index=True)

    # Оценка модели
    evaluate_model(y_test, y_predicted, len(y_test), X_test_mn.shape[1] - 1, model)
