import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import random 

def segment_intersects_cell(r, c, N, x0, y0, x1, y1):
    cx0, cx1 = c / N, (c + 1) / N
    cy0, cy1 = r / N, (r + 1) / N
    dx, dy = x1 - x0, y1 - y0
    tmin, tmax = 0.0, 1.0

    for p, q in [(-dx, x0 - cx0), (dx, cx1 - x0),
                 (-dy, y0 - cy0), (dy, cy1 - y0)]:
        if abs(p) < 1e-10:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                tmin = max(tmin, t)
            else:
                tmax = min(tmax, t)
        if tmin > tmax:
            return False

    return (tmax - tmin) > 1e-10

def create_lines(N):
    global LINES
    LINES = []  # start fresh each time
    for i in range(N):
        # Pick a random edge to start from (left, right, top, bottom)
        edge = random.choice(['left', 'right', 'top', 'bottom'])
        if edge == 'left':
            start_x, start_y = 0.0, random.random()
            end_x,   end_y   = 1.0, random.random()
        elif edge == 'right':
            start_x, start_y = 1.0, random.random()
            end_x,   end_y   = 0.0, random.random()
        elif edge == 'top':
            start_x, start_y = random.random(), 0.0
            end_x,   end_y   = random.random(), 1.0
        else:  # bottom
            start_x, start_y = random.random(), 1.0
            end_x,   end_y   = random.random(), 0.0

        LINES.append((start_x, start_y, end_x, end_y))

def create_results(N, total=100000):

    parts = [random.randint(0,100) for _ in range(N * N)]
    sum_parts = sum(parts)
    normalized = [p/sum_parts * total for p in parts]
    rounded = [int(round(p)) for p in normalized ]
    diff = total - sum(rounded)
    rounded[-1] += diff

    results = np.zeros((N * N, 1))
    for i in range(len(rounded)):
        results[i, 0] = rounded[i]
    return results
        



def subdivide_square(N, lines):
    grid = np.zeros((N, N), dtype=int)
    per_line = []
    global square_density
    for (x0, y0, x1, y1) in lines:
        flat = []
        for r in range(N):
            for c in range(N):
                hit = segment_intersects_cell(r, c, N, x0, y0, x1, y1)
                flat.append(1 if hit else 0)
                if hit:
                    grid[r, c] = 1
        per_line.append(np.array(flat))

    combined = np.array(per_line)


    combined_unique, unique_indices, inverse = np.unique(combined, axis=0, return_index=True, return_inverse=True)
    results_full = create_results(N)

    # Make duplicates share the same result as the first occurrence
    results_deduped = results_full[unique_indices][inverse]  # maps every row back to its unique result
    results = results_deduped[unique_indices]                # trim to unique rows
    # Solve ("_" just means i dont care about these values and dont store them, still needed to unpack)
    values, _, _, _ = np.linalg.lstsq(combined_unique, results, rcond=None)

    # Zero out cells never intersected by any line
    cell_sums = combined.sum(axis=0)
    values[cell_sums == 0] = 0

    square_density = values.reshape(N,N)
    print(values)
    eiganvalues, _ = np.linalg.eig(combined_unique.T @ combined_unique)

    #print("eiganvalues:", eiganvalues)
    return grid, per_line, combined, values.flatten(), eiganvalues

#LINES = [
 #   (0.0, 1/8, 1.0, 1/8), # w
  #  (0.0, 5/8, 1.0, 5/8),   # horizontal at 3/8 from bottom x
   # (0.0, 0.0, 1.0, 1.0),   # diagonal y
    #(1/8, 0, 1/8, 1.0), # z
#]


def draw(ax, N):
    ax.clear()
    grid, per_line, combined, values, eiganvalues = subdivide_square(N, LINES)
    emin, emax = eiganvalues.min(), eiganvalues.max()
    cnumber = emax / emin
    vmin, vmax =  values.min(), values.max() # needed for normalisation
    for r in range(N):
        for c in range(N):
            cell_val = values[ r * N + c] #unsure
            cell_eigan = eiganvalues[r * N + c]
            if grid[r,c]: 
                norm = (cell_val - vmin) / (vmax - vmin + 1e-10) #normalises values
                color = plt.cm.viridis(norm) #sets colour based on built in matplotlib color set
            else: 
                color = '#F1EFE8'
            rect = patches.Rectangle(
                (c / N, 1 - (r + 1) / N),
                1 / N, 1 / N,
                linewidth=0.5,
                edgecolor='#B4B2A9',
                facecolor=color
            )
            ax.add_patch(rect)

            if grid[r, c]:
                ax.text(
                (c + 0.5) / N,
                1 - (r + 0.5) / N,
                f'{cell_val:.0f} \n {cell_eigan}',
                ha='center', va='center',
                fontsize=max(4, 8 - N // 4),  # shrinks text as grid grows
                color='white' if norm < 0.0 else 'black'  # contrast against background
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{N}×{N} grid, there is {len(LINES)} lines, condition number = {cnumber}', fontsize=10)
    ax.axis('off')


fig, ax = plt.subplots(figsize=(5, 5))
plt.subplots_adjust(bottom=0.15)
create_lines(N=2*2)
draw(ax, N=2)

slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(slider_ax, 'Grid size', 2, 100, valinit=2, valstep=1)

slider_ay = fig.add_axes([0.2, 0.05, 0.6, 0.03])
slidery = Slider(slider_ay, 'Line Alpha', 0, 1, valinit=0.05, valstep=0.01)
def on_change(val):
    N = int(slider.val)
    alpha = slidery.val  # read from slider
    create_lines(N * N)
    fig.canvas.draw_idle()
    draw(ax, N)

    for (x0, y0, x1, y1) in LINES:
        ax.plot([x0, x1], [1 - y0, 1 - y1], color='black', linewidth=1.5, label='', alpha=alpha)

slider.on_changed(on_change)
slidery.on_changed(on_change)  # redraw when alpha changes too
plt.show()
