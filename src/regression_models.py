import numpy as np
import matplotlib.pyplot as plt
from src.visualization import save_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regressor(regressor, X_train, X_test, y_train, y_test, model_name):
    """Обучает регрессор, вычисляет метрики и сохраняет графики."""
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.2f}")

    # Визуализация предсказаний
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.random.choice(len(y_test), size=min(len(y_test), 100), replace=False)
    ax.scatter(indices, y_test.iloc[indices], color='blue', label='Фактические значения')
    ax.scatter(indices, y_pred[indices], color='red', label='Предсказания')
    ax.set_xlabel('Индекс')
    ax.set_ylabel('Цена')
    ax.set_title(f'Фактические и предсказанные значения ({model_name})')
    ax.legend()

    save_plot(fig, f"{model_name}_predictions.png")
    plt.close(fig)