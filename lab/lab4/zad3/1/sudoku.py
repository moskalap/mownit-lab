import random

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def build_sudoku_grid(lines, k):
    unmovalbe = np.zeros((9,9)).astype(bool)
    blank_spaces_cnt = 0
    blank_spaces_map = {}
    grid = np.zeros((9, 9)).astype(int)
    if len(lines) != 9:
        raise IndexError('Sudoku has to be 9x9')

    for i in range(0, 9):
        blank_spaces_map[i] = []
        line_splitted = lines[i][0:9]
        if len(line_splitted) != 9:
            raise IndexError('Sudoku has to be 9x9')
        for j in range(0, 9):
            if line_splitted[j] == '0':
                blank_spaces_map[i].append(j)
                blank_spaces_cnt += 1
            else:
                grid[i, j] = int(line_splitted[j])
                unmovalbe[i,j] = True
    plot_sudoku(grid, k+'-0')
    for i in range(0,3):
        for j in range(0,3):
            fullfill(grid, 3*i,(i+1)*3, 3*j, (j+1)*3)

    return grid, blank_spaces_cnt, unmovalbe

def fullfill(arr,i1,i2,j1,j2):
    to_add = [x for x in [y for y in range(0,10)] if x not in arr[i1:i2,j1:j2].ravel()]
    print(to_add)
    print(arr[i1:i2,j1:j2])
    ind = 0
    for i in range(i1,i2):
        for j in range(j1,j2):
            if arr[i,j] == 0:
                arr[i,j] = to_add[ind]
                ind += 1


def swap(unmovable, sudoku_grid):
    ib, jb = 3 * np.random.randint(3, size=2)
    row1, col1 = np.random.randint(3, size=2)
    while unmovable[ib + row1, jb + col1]:
        row1, col1 = np.random.randint(3, size=2)
    row2, col2 = np.random.randint(3, size=2)
    while unmovable[ib + row2, jb + col2] or (row1 == row2 and col1 == col2):
        row2, col2 = np.random.randint(3, size=2)

    row2 += ib
    row1 += ib
    col1 += jb
    col2 += jb

    temp = sudoku_grid[row1, col1]
    sudoku_grid[row1, col1] = sudoku_grid[row2,col2]
    sudoku_grid[row2,col2] = temp
    #print('from ({}, {}) to  ({}, {})'.format(row1,col1,row2,col2))
    return row1,col1, row2, col2

def rollback_swap(sudoku_grid, row1, col1, row2, col2):
    temp = sudoku_grid[row1, col1]
    sudoku_grid[row1, col1] = sudoku_grid[row2, col2]
    sudoku_grid[row2, col2] = temp




def sim_anneal(sudoku_grid, unmovable, temp_start, temp_end, temp_fn):
    best_energy = count_energy(sudoku_grid)
    iters = 0
    static = 0
    while temp_start > temp_end and best_energy > 0 :
        x1, y1, x2, y2 = swap(unmovable, sudoku_grid)
        iters += 1
        static += 1
        if static == 200000:
            temp_start+=2
        temp_start = temp_fn(temp_start)
        new_energy = count_energy(sudoku_grid)
        try:

            prop = np.math.exp((best_energy - new_energy) / max(0.004, temp_start))
        except OverflowError as e:
            prop = 0

        if iters % 10000 == 0:

            print('{} : prob {}, temp: {}, energy {}, row {}, col {}, bl {} '.format(iters, prop, temp_start, new_energy, rpt_in_row(sudoku_grid), rpt_in_col(sudoku_grid), rpt_in_blcks(sudoku_grid)))
        if new_energy == 0:
            break
        if new_energy < best_energy:
            best_energy = new_energy
            static = 0

        else:

            if prop < random.random():
                rollback_swap(sudoku_grid, x1, y1, x2, y2)
            else:
                best_energy = new_energy

    print('{} : prob {}, temp: {}, energy {}, row {}, col {}, bl {} '.format(iters, prop, temp_start, new_energy,
                                                                             rpt_in_row(sudoku_grid),
                                                                             rpt_in_col(sudoku_grid),
                                                                             rpt_in_blcks(sudoku_grid)))
    return iters

def count_energy(sudoku_grid):
    return rpt_in_row(sudoku_grid) + rpt_in_col(sudoku_grid) + rpt_in_blcks(sudoku_grid)


def rpt_in_blck(block):
    return 9 - len(Counter(block.ravel()))


def rpt_in_blcks(sudoku_grid):
    sum_of_blcks = 0
    for i in range(0,3):
        for j in range(0,3):
            sum_of_blcks += rpt_in_blck(sudoku_grid[ 3*i: (i+1)*3, 3*j: (j+1)*3])
    return sum_of_blcks


def rpt_in_col(sudoku_grid):
    rpt_sum = 0
    for col in range(0, 9):
        rpt_sum += 9 - len(Counter(sudoku_grid[:, col]))
    return rpt_sum



def rpt_in_row(sudoku_grid):
    rpt_sum = 0
    for row in range(0,9):
        rpt_sum += 9 - len(Counter(sudoku_grid[row,:]))
    return rpt_sum


def solve_sudoku(sudoku_grid, unmovable):
    print_sudoku(sudoku_grid)
    iterations = sim_anneal(sudoku_grid, unmovable, 400, 0.001, lambda x: x*0.9999)
    en = count_energy(sudoku_grid)
    return sudoku_grid, iterations,en

def print_sudoku(sudoku_grid):
    for i in range(0,9):
        line_str = ''
        for j in range(0,9):
            if j % 3 == 0:
                line_str += '\t'
            line_str += str(sudoku_grid[i,j])+' '
        if i % 3 == 0:
            print('\n')
        print(line_str)


def plot_sudoku(n, name):
    # Simple plotting statement that ingests a 9x9 array (n), and plots a sudoku-style grid around it.
    plt.figure()
    for y in range(10):
        plt.plot([-0.05, 9.05], [y, y], color='black', linewidth=1)

    for y in range(0, 10, 3):
        plt.plot([-0.05, 9.05], [y, y], color='black', linewidth=3)

    for x in range(10):
        plt.plot([x, x], [-0.05, 9.05], color='black', linewidth=1)

    for x in range(0, 10, 3):
        plt.plot([x, x], [-0.05, 9.05], color='black', linewidth=3)

    plt.axis('image')
    plt.axis('off')  # drop the axes, they're not important here

    for x in range(9):
        for y in range(9):
            foo = n[8 - y][x]  # need to reverse the y-direction for plotting
            if foo > 0:  # ignore the zeros
                T = str(foo)
                plt.text(x + 0.3, y + 0.2, T, fontsize=20)

    plt.savefig(name+'.jpg', dpi = 96)


if __name__ == '__main__':
    with open('./res/sudoku.txt') as f:
        line = f.readlines()
        sudokus = {}
        sudoku_l = None
        for l in line:
            if l.startswith('Grid'):
                if sudoku_l != None:
                    sudokus[l] = sudoku_l
                sudoku_l = []
            else:
                sudoku_l.append([i for i in l])
        print(len(sudokus))

        for k in sudokus.keys():
            sudoku_grid, blank_spaces, unmovable = build_sudoku_grid(sudokus[k], k)
            solved_sudoku, iterations, en= solve_sudoku(sudoku_grid, unmovable)
            if en ==0:
                plot_sudoku(solved_sudoku, k+str(iterations)+'solv')
            else:
                plot_sudoku(solved_sudoku, k + str(iterations))