import pandas as pd

def predict_with_user_input(model, scaler):
    """
    Позволяет пользователю ввести данные через консоль для предсказания стоимости недвижимости.

    Параметры:
    - model: обученная модель для предсказаний.
    - scaler: объект StandardScaler для масштабирования пользовательских данных.
    """
    while True:
        print("\nВведите данные для расчета стоимости недвижимости:")
        user_input = []
        feature_names = ['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year', 'city_Kraków', 'city_Poznañ', 'city_Warszawa']
        prompts = [
            "Этаж (целое число, например, 5): ",
            "Широта (например, 52.13): ",
            "Долгота (например, 21.02): ",
            "Количество комнат (целое число, например, 4): ",
            "Площадь (например, 100): ",
            "Год постройки (например, 2020): ",
            "Город Краков (1, если да; 0, если нет): ",
            "Город Познань (1, если да; 0, если нет): ",
            "Город Варшава (1, если да; 0, если нет): "
        ]

        for i, prompt in enumerate(prompts):
            value = input(prompt)
            try:
                if feature_names[i] in ['floor', 'rooms', 'year', 'city_Kraków', 'city_Poznañ', 'city_Warszawa']:
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                print(f"Некорректное значение для {feature_names[i]}. Попробуйте ещё раз.")
                break
            user_input.append(value)

        if len(user_input) == len(feature_names):
            # Преобразование в DataFrame с именами признаков
            user_input_df = pd.DataFrame([user_input], columns=feature_names)
            user_input_scaled = scaler.transform(user_input_df)
            prediction = model.predict(user_input_scaled)[0]
            print(f"\nПредсказанная стоимость недвижимости: {round(prediction)}$")
        else:
            print("Ошибка ввода. Попробуйте снова.")

        continue_prompt = input("\nХотите попробовать ещё раз? (да/нет): ").strip().lower()
        if continue_prompt not in ['да', 'yes', 'y']:
            print("Завершение работы.")
            break