from src.data_processing import load_data, preprocess_data
from src.visualization import plot_correlation_matrix
from src.regression_models import evaluate_regressor
from src.utils import setup_logging, log_message
from src.user_input import predict_with_user_input

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

def main():
    setup_logging()
    log_message("Начало работы приложения.")

    # Загрузка и обработка данных
    file_path = './data/houses.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Визуализация
    plot_correlation_matrix(df)

    # Подготовка данных для обучения
    X = df[['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year', 'city_Kraków', 'city_Poznañ', 'city_Warszawa']]
    y = df['price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Регрессоры
    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "PassiveAggressive": PassiveAggressiveRegressor(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    # Обучение и оценка всех регрессоров
    for name, regressor in regressors.items():
        log_message(f"Обучение модели {name}")
        evaluate_regressor(regressor, X_train, X_test, y_train, y_test, name)

    # Использование CatBoostRegressor для ввода данных пользователем
    log_message("Ввод пользовательских данных для CatBoostRegressor")
    predict_with_user_input(regressors["CatBoost"], scaler)

    log_message("Работа приложения завершена.")

if __name__ == "__main__":
    main()