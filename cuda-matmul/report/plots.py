import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


def tile():
    df = pd.read_csv('tile-results.csv', sep=';')

    df_k = df[df['TILE_K'] == 32]
    pivot = df_k.pivot_table(index='TILE_M', columns='TILE_N', values='ms', aggfunc="mean")
    
    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()

    # Вычисляем границы клеток (в логарифмическом масштабе)
    def compute_edges(vals):
        vals = np.array(vals)
        edges = np.sqrt(vals[:-1] * vals[1:])  # геометрические середины
        # дополняем левый и правый край
        left  = vals[0]**2 / vals[1]
        right = vals[-1]**2 / vals[-2]
        return np.concatenate([[8**0.5], edges, [32 * 2**0.5]])

    x_edges = compute_edges(x_vals)
    y_edges = compute_edges(y_vals)

    # Построение графика
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(x_edges, y_edges, pivot.values, shading='flat')

    # логарифмические шкалы
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # настраиваем тики
    ticks = [4, 8, 16, 32]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    ax.set_xlabel('TILE_N')
    ax.set_ylabel('TILE_M')
    fig.colorbar(pcm, ax=ax, label='ms')
    plt.savefig('tile_heatmap.png')

    pairs = df[['TILE_M', 'TILE_N']].drop_duplicates()
    plt.figure(figsize=(8, 6))
    for m, n in pairs.itertuples(index=False):
        df_mn = df[(df['TILE_M'] == m) & (df['TILE_N'] == n)].sort_values('TILE_K')
        plt.plot(df_mn['TILE_K'], df_mn['ms'], marker='o', label=f'M={m}, N={n}')
    plt.title('Execution Time (ms) for TILE_K')
    plt.xlabel('TILE_K')
    plt.ylabel('ms')
    plt.legend()
    plt.grid(True)
    plt.savefig('tile_graphic.png')

    min_ms = df['ms'].min()
    best = df[df['ms'] == min_ms]
    print(f"Minimal execution time: {min_ms} ms")
    print("Best configuration):")
    print(best[['TILE_M', 'TILE_N', 'TILE_K', 'ms']].to_string(index=False))
    
def vector():
    df = pd.read_csv('vector-results.csv', sep=';')

    df['vec_factor'] = df['VEC_X'] * df['VEC_Y']
    df['tile_cfg'] = df['TILE_M'].astype(str) + 'x' + \
                    df['TILE_N'].astype(str) + 'x' + df['TILE_K'].astype(str)

    plt.figure(figsize=(8, 6))
    for tile in sorted(df['tile_cfg'].unique()):
        sub = df[df['tile_cfg'] == tile].sort_values('vec_factor')
        if sub.empty:
            continue
        plt.plot(sub['vec_factor'], sub['ms'], marker='o', label=tile)
    plt.xlabel('VEC_X * VEC_Y')
    plt.ylabel('ms')
    plt.title('Execution Time for Vectorization')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.savefig('vector_graphic.png')

    sub = df[df['TILE_K'] == 32]
    pivot = sub.pivot_table(index='vec_factor', columns='TILE_M', values='ms', aggfunc='mean')

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, aspect='auto', origin='lower')
    plt.colorbar(label='Execution Time (ms)')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('TILE_M')
    plt.ylabel('VEC_X * VEC_Y')
    plt.title(f'Heatmap (TILE_K=32)')
    plt.savefig('vector_heatmap.png')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(pivot.columns.astype(float), pivot.index.astype(float))
    Z = pivot.values
    ax.plot_surface(X, Y, Z, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('TILE_M')
    ax.set_ylabel('VEC_X * VEC_Y')
    ax.set_zlabel('ms')
    ax.set_title(f'3D Surface (TILE_K=32)')
    plt.savefig('surface.png')

    min_ms = df['ms'].min()
    best = df[df['ms'] == min_ms]
    print(f"Minimal execution time: {min_ms} ms")
    print("Best configuration):")
    print(best[['VEC_X', 'VEC_Y', 'TILE_M', 'TILE_N', 'TILE_K', 'ms']].to_string(index=False))


def sizes():
    df = pd.read_csv('sizes-results2.csv', sep=';')

    plt.figure(figsize=(8, 6))
    plt.plot(df['size'], df['1'], marker='o', label="tiled")
    plt.plot(df['size'], df['2'], marker='o', label="vectorized")
    plt.plot(df['size'], df['3'], marker='o', label="tiled-padded")
    plt.plot(df['size'], df['4'], marker='o', label="vectorized-padded")
    plt.xlabel('data size')
    plt.ylabel('ms')
    plt.title('Execution Time for different input data size')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.savefig('sizes2.png')

vector()
tile()
sizes()
plt.show()
