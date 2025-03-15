import pandas as pd

def load_data(file_path):
    """Загружает данные из CSV и выполняет начальную обработку."""
    df = pd.read_csv(file_path, encoding='latin1')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

def preprocess_data(df):
    """Обрабатывает данные (фильтрация, преобразование, удаление выбросов)."""
    # Конвертация цен в доллары
    conversion_rate = 0.26
    df['price'] = df['price'] * conversion_rate
    df = df[df['price'] >= 5000]

    # Фильтрация площади
    df = df[(df['sq'] >= 15) & (df['sq'] < 1000)]

    # Фильтрация года
    df = df[(df['year'] >= 1940) & (df['year'] <= 2025)]

    # Добавление столбца цены за кв.м.
    df['price_per_sq'] = df['price'] / df['sq']

    # Удаление выбросов
    for col in ['price_per_sq', 'year', 'sq']:
        q_low = df[col].quantile(0.001)
        q_hi = df[col].quantile(0.999)
        df = df[(df[col] < q_hi) & (df[col] > q_low)]

    # Преобразование городов в dummy-переменные
    df = pd.get_dummies(df, columns=['city'])
    df = df.drop(['address', 'id'], axis=1)

    return df