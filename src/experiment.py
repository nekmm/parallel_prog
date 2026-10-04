import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SIZES = [100, 300, 500, 800, 1000]
THREADS = [1, 2, 4, 8]

def generate_data(size):
    fileA = f"data/matrix_A_{size}.txt"
    fileB = f"data/matrix_B_{size}.txt"
    if not os.path.exists(fileA):
        print(f"Генерация матриц размера {size}...")
        A = np.random.rand(size, size) * 10
        B = np.random.rand(size, size) * 10
        np.savetxt(fileA, A, fmt='%.4f')
        np.savetxt(fileB, B, fmt='%.4f')

def run_experiments():
    results = []
    binary = "./matrix_mult.exe" if os.name == "nt" else "./matrix_mult"
    if not os.path.exists(binary):
        print("Компиляция C++ программы...")
        subprocess.run(["g++", "-O3", "-fopenmp", "src/matrix_mult.cpp", "-o", "matrix_mult"], check=True)

    for size in SIZES:
        generate_data(size)
        
        for threads in THREADS:
            print(f"Запуск: Размер={size}, Потоков={threads}")
            
            result = subprocess.run(
                [binary, str(size), str(threads)],
                capture_output=True, text=True
            )
            try:
                output = result.stdout.strip().split(',')
                if len(output) == 4:
                    n = int(output[0])
                    t = int(output[1])
                    time_sec = float(output[2])
                    ops = int(output[3])
                    
                    # Защита от деления на ноль
                    if time_sec > 0:
                        gflops = (2 * ops) / (time_sec * 1e9)
                    else:
                        gflops = 0.0

                    results.append({
                        "Size": n,
                        "Threads": t,
                        "Time_sec": time_sec,
                        "Operations": ops,
                        "GFLOPS": gflops          # <-- используем переменную gflops
                    })
            except Exception as e:
                print(f"Ошибка парсинга: {e}, вывод: {result.stdout}")

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/experiment_results.csv", index=False)
    print("\nРезультаты сохранены в results/experiment_results.csv")
    
    return df
def plot_results(df):
    os.makedirs("results/plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for threads in df['Threads'].unique():
        subset = df[df['Threads'] == threads]
        plt.plot(subset['Size'], subset['Time_sec'], marker='o', label=f'{threads} поток(ов)')
    
    plt.title('Зависимость времени выполнения от размера матрицы')
    plt.xlabel('Размер матрицы (N x N)')
    plt.ylabel('Время выполнения (сек)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/time_vs_size.png')
    plt.close()

    max_size = df['Size'].max()
    subset = df[df['Size'] == max_size]
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset['Threads'], subset['Time_sec'], marker='s', color='red')
    plt.title(f'Зависимость времени выполнения от количества потоков (N={max_size})')
    plt.xlabel('Количество потоков')
    plt.ylabel('Время выполнения (сек)')
    plt.xticks(subset['Threads'])
    plt.grid(True)
    plt.savefig('results/plots/time_vs_threads.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    base_time = subset[subset['Threads'] == 1]['Time_sec'].values[0]
    speedup = base_time / subset['Time_sec']
    plt.plot(subset['Threads'], speedup, marker='^', color='green', label='Ускорение')
    plt.plot(subset['Threads'], subset['Threads'], '--', color='gray', label='Идеальное ускорение')
    plt.title(f'Ускорение от распараллеливания (N={max_size})')
    plt.xlabel('Количество потоков')
    plt.ylabel('Ускорение (S = T1 / Tn)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/speedup.png')
    plt.close()

    print("Графики сохранены в results/plots/")

if __name__ == "__main__":
    df = run_experiments()
    plot_results(df)