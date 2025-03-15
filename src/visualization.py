import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(fig, filename):
    """Сохраняет график в папку outputs."""
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    print(f"График сохранен: {file_path}")

def plot_correlation_matrix(df):
    """Строит и сохраняет матрицу корреляций."""
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Матрица корреляций")
    save_plot(fig, "correlation_matrix.png")
    plt.close(fig)