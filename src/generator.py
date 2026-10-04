import numpy as np
import os
import sys

# Создаем папку data при запуске скрипта
os.makedirs("data", exist_ok=True)

def generate_matrix(size, prefix):
    filepath = f"data/matrix_{prefix}_{size}.txt"
    
    if os.path.exists(filepath):
        print(f"Файл уже существует: {filepath}, пропускаем.")
        return
    
    matrix = np.random.rand(size, size) * 10
    np.savetxt(filepath, matrix, fmt='%.4f')
    print(f"Матрица {size}x{size} сохранена в {filepath}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1:]]
    else:
        sizes = [100, 300, 500, 800, 1000]
    for N in sizes:
        generate_matrix(N, "A")
        generate_matrix(N, "B")
    
    print("\nГенерация завершена!")