from threading import Thread

import numpy as np
import math
import random
ret = 0.9999
def read_neighoburs(param):
    neighbours = {}
    i = -1
    with open(param) as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line == '.':
                i+=1
                neighbours[i] = []
            else:
                neighbours[i].append(line)
    print(neighbours)

    for k in neighbours.keys():
        neighbour = neighbours[k]
        nieghbours_list = []
        R = len(neighbour)
        C = len(neighbour[0])
        print('neighbour {} r: {} c:{}'.format(neighbour, R,C))
        for r in range(0, R):
            for c in range(0, C):
                if neighbour[r][c] == 'x':
                    x, y = r,c
        for r in range(0, R):
            for c in range(0, C):
                if neighbour[r][c] == '#':
                    a = r-x
                    b = c-y
                    neighbour_eq = (a,b)

                    print('lamda x + {} ,y+ {}'.format(a,b))
                    nieghbours_list.append(neighbour_eq, )
        neighbours[k] = (nieghbours_list, len(neighbour))

    return neighbours
def mark_neighbours(bit_image, x, y, neighobur):
    for neigbour_fun in neighobur:
        a, b = neigbour_fun
        print(x+a,y+b)
        bit_image[x+a,y+b] = True
def count_energy(bit_image, x, y, n, neigbour):
    energy = 0
    for neighbour_ratios in neigbour:
        a,b = neighbour_ratios
        nx = (x + a)
        ny = (y + b)

        if nx < n and ny < n and nx > -1 and ny > - 1 and bit_image[nx,ny]:
            energy+=1
    return energy
def count_energy_in_range(bit_image, start, stop, neighbour, id, results):
    partial_energy = 0
    x1,y1 = start
    x2,y2 = stop
    for x in range(x1,x2):
        for y in range(y1,y2):
            partial_energy += count_energy(bit_image,x,y,n,neighbour)
    results[id] = partial_energy

def count_full_energy(bit_image, n, neighbour):
    m = n//4


    start1 = (0,0)
    stop1 = (m,m)

    start2 = (m+1,0)
    stop2 =  (n,m)

    start3 = (0, m+1)
    stop3 = (m, n)

    start4 =(m+1,n)
    stop4 =(m+1,n)
    results = {}
    threads = []
    threads.append(Thread(target=count_energy_in_range, args =(np.copy(bit_image), start1, stop1, neighbour, 1, results)))
    threads.append(Thread(target=count_energy_in_range, args =(np.copy(bit_image), start2, stop2, neighbour, 2, results)))
    threads.append(Thread(target=count_energy_in_range, args =(np.copy(bit_image), start3, stop3, neighbour, 3, results)))
    threads.append(Thread(target=count_energy_in_range, args =(np.copy(bit_image), start4, stop4, neighbour, 4, results)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()



    full_energy = 0
    for x in results.values():
        full_energy += x
    return full_energy

def swap(bit_image, n):
    while True:
        x1 = np.random.randint(0,n)
        x2 = np.random.randint(0,n)

        y1 = np.random.randint(0,n)
        y2 = np.random.randint(0,n)

        if bit_image[x1,y1] != bit_image[x2,y2]:
            bit_image[x1,y1] = not bit_image[x1,y1]
            bit_image[x2,y2] = not bit_image[x2,y2]
            return (x1,y1),(x2,y2)

def visualize(snapshots, neighbour, neighbour_name, iters, vals):
    import os
    if not os.path.exists('./'+str(neighbour_name)):
        os.makedirs('./'+str(neighbour_name))
    import matplotlib.pyplot as plt
    for snapshot in snapshots:
        iter, val, img = snapshot
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.subplot(2, 1, 2)


        plt.plot(iters, vals, 'ko-', iter, val, 'ro')
        plt.title('iteracje')
        plt.ylabel('Wartość energii')
        plt.savefig('./' + str(neighbour_name) + '/iter' + str(iter) + '_en.png', dpi=96)
        plt.clf()

        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.savefig('./'+str(neighbour_name)+'/iter'+str(iter)+'.png', dpi=326)




def rollback_swap(bit_image, swapped):
    (x1,y1), (x2,y2) = swapped
    bit_image[x1, y1] = not bit_image[x1, y1]
    bit_image[x2, y2] = not bit_image[x2, y2]


def build_energy_map(bit_image, n, neighbour_def):
    energy_map = np.zeros(n)
    for i in range(0,n):
        sum_of_row = 0
        for j in range(0,n):
            sum_of_row += count_energy(bit_image,i,j,n,neighbour_def)
        energy_map[i] = sum_of_row
    print(sum(energy_map))
    return energy_map
def count_energy_for_row(bit_image, row, n, neighbour_def):
    sum_of_row = 0
    for c in range(0, n):
        sum_of_row += count_energy(bit_image,row,c,n,neighbour_def)
    return sum_of_row


def generate_binary_image(n, r):
    bit_image = np.zeros((n, n), dtype='bool')
    i = 0
    while i < n * n * r:
        x = np.random.randint(0, n)
        y = np.random.randint(0, n)
        if not bit_image[x, y]:
            bit_image[x, y] = True
            i += 1

    return bit_image


def count_energy_from_map(binary_image, n, energy_map, swapped,neighbour_def, neighbour_len):


    (x1,y1), (x2,y2) = swapped
    a1 = max(x1 - neighbour_len, 0)
    b1 = min(x1 + neighbour_len+1, n)
    a2 = max(x2 - neighbour_len, 0)
    b2 = min(x2 + neighbour_len+1, n)
    a = range(a1, b1)
    b = range(a2, b2)
    sum_energy = 0
    energies = {}
    for k in range (0, n):
        if k in a or k in b:
            en = count_energy_for_row(binary_image, k, n, neighbour_def)
            energies[k] = en
            sum_energy += en
        else:
            sum_energy += energy_map[k]

    return sum_energy, energies


def update_energy_map(energy_map, energy_map_rows_to_change):
    for k in energy_map_rows_to_change.keys():
        energy_map[k] = energy_map_rows_to_change[k]



if __name__ == "__main__":
    n = 512
    neighbours = read_neighoburs('neighbours.txt')
    for neighbour_k in neighbours.keys():
        for r in [0.1, 0.2, 0.5, 0.6, 0.8]:
            neighbour_def, neighbour_len = neighbours[neighbour_k]
            folder_name = 'neigh{}r{}'.format(neighbour_k,r)
            binary_image = generate_binary_image(n, r)

            temp = 400
            stop_temp = 0.001
            end_iter = 2000000
            iter = 0
            iters = []
            energies = []
            snapshots = []


            energies_change = 0
            energy_map = build_energy_map(binary_image,n, neighbour_def)
            best_energy = sum(energy_map)

            while temp > stop_temp and iter < end_iter:
                iter += 1


                swapped = swap(binary_image, n)
                new_energy, energy_map_rows_to_change = count_energy_from_map(binary_image,n, energy_map, swapped,neighbour_def, neighbour_len)
                #print(new_energy)
                if iter % 1000 == 0:
                    print('{} : prob {}, temp: {}, iter {}, energy {}'.format(folder_name, math.exp((best_energy - new_energy) / temp),
                                                                         temp, iter, best_energy))
                if new_energy < best_energy:
                    best_energy = new_energy
                    energies_change += 1
                    update_energy_map(energy_map, energy_map_rows_to_change)
                else:

                    if math.exp((best_energy - new_energy) / temp) < random.random():
                        rollback_swap(binary_image, swapped)

                    else:
                        best_energy = new_energy
                        update_energy_map(energy_map, energy_map_rows_to_change)

                if energies_change % 100 == 0:
                    energies_change += 1
                    print('snapshot')
                    snapshots.append((iter, best_energy, np.copy(binary_image)))


                temp *= ret
                iters.append(iter)
                energies.append(best_energy)



            snapshots.append((iter, best_energy, binary_image))

            visualize(snapshots, neighbour_def,folder_name,iters,energies)

















