import numpy as np
import os
import sys

def load_matrix(filepath):
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл не найден: {filepath}")
        sys.exit(1)
    return np.loadtxt(filepath)

def verify():
    N = 1000        
    THREADS = 4    

    file_A = f"data/matrix_A_{N}.txt"
    file_B = f"data/matrix_B_{N}.txt"
    file_C_cpp = f"results/matrix_C_{N}_{THREADS}.txt"

    print("=" * 50)
    print("Верификация результатов")
    print("=" * 50)
    print(f"Размер матрицы: {N}x{N}")
    print(f"Количество потоков: {THREADS}")
    print(f"Файл A: {file_A}")
    print(f"Файл B: {file_B}")
    print(f"Результат C++: {file_C_cpp}")
    print("-" * 50)

    print("Загрузка исходных матриц")
    A = load_matrix(file_A)
    B = load_matrix(file_B)
    
    print("Загрузка результата C++ программы")
    C_cpp = load_matrix(file_C_cpp)

    if A.shape != (N, N) or B.shape != (N, N):
        print(f"Ошибка: Размеры исходных матриц не совпадают с ожидаемыми ({N}x{N}).")
        print(f"   A: {A.shape}, B: {B.shape}")
        sys.exit(1)
        
    if C_cpp.shape != (N, N):
        print(f"Ошибка: Размер результата C++ не совпадает с ожидаемым ({N}x{N}).")
        print(f"   C_cpp: {C_cpp.shape}")
        sys.exit(1)

    print("Вычисление эталонного результата с помощью NumPy")
    C_python = np.dot(A, B)

    print("Сравнение результатов...")

    is_correct = np.allclose(C_cpp, C_python, rtol=1e-5, atol=1e-8)

    if is_correct:
        print("\nВерификация пройдена успешно!")
        print("Результаты C++ и NumPy совпадают в пределах допустимой погрешности.")
    else:
        print("\nВерификация не пройдена!")
        print("Результаты различаются. Возможные причины:")
        print("  1. Ошибка в алгоритме C++ (неверный порядок циклов, выход за границы).")
        print("  2. Повреждены входные или выходные файлы.")
        print("  3. Слишком большая погрешность вычислений (маловероятно для double).")
        
        diff = np.abs(C_cpp - C_python)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"\nСтатистика расхождений:")
        print(f"  Максимальное расхождение: {max_diff:.6e}")
        print(f"  Среднее расхождение:     {mean_diff:.6e}")
        
        print("\nПримеры расхождений (первые 5):")
        indices = np.argwhere(~np.isclose(C_cpp, C_python, rtol=1e-5, atol=1e-8))
        for idx in indices[:5]:
            i, j = idx
            print(f"  C[{i}][{j}]: C++ = {C_cpp[i, j]:.6f}, Python = {C_python[i, j]:.6f}, Diff = {abs(C_cpp[i, j] - C_python[i, j]):.6e}")
        
        sys.exit(1)

    print("=" * 50)

if __name__ == "__main__":
    verify()