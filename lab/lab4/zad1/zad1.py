import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
a = 0.9999
snapshots = {}
out = open('out.tex', 'a+')


def swap(t, N):
    i = random.randint(0,N-1)
    j = random.randint(0,N-1)
    temp = t[i]
    t[i] = t[j]
    t[j] = temp
    return i,j

def rollback_swap(t,i,j):
    temp = t[i]
    t[i] = t[j]
    t[j] = temp

def count_distance(t, distances):
    sum = 0
    for i in range(0, len(t)):
        sum += distances[t[i], t[(i+1) % len(t)]]
    return sum

def make_a_snapshot(iter, current_distance, current_path):
    copy_of_path = list(current_path)
    snapshots[iter] = (current_distance,copy_of_path
                       )


def make_plots(iters, distances, snapshots, points, solution, best_distance):


    for k in sorted(snapshots.keys()):


        plt.clf()

        plt.subplot(2, 1, 1)
        plt.plot(iters, distances, 'ko-')
        plt.title('Wartoci')
        plt.ylabel('Dlugosc sciezki')

        plt.subplot(2, 1, 2)
        (current_distance, current_path) = snapshots[k]
        p = np.array([points[i] for i in current_path])

        plt.scatter(p[:,0], p[:,1])
        plt.plot(p[:, 0], p[:, 1])
        plt.savefig('img/'+str(k)+'.png')

def generate_point_linear(N):
    points = np.random.rand(N, 2)*500

    return points


def generate_point_normal(N, center= 50, nercenter = 30, neardge = 15, edge = 5):
    in_center = int(N*center/100)
    in_nercener = int(N*nercenter/100)
    in_nearedge = int(N*neardge/100)
    in_edge = int(N*edge/100)

    edge_p = np.random.rand(in_edge,2)*500
    nearedge_p = ((np.random.rand(in_nearedge,2)*500) % 400) +50

    nearcenter_p = ((np.random.rand(in_nercener,2)*500) % 300) +100
    center_p = ((np.random.rand(in_center,2)*500) % 200) + 150
    points = np.concatenate((edge_p, nearedge_p, nearcenter_p, center_p), axis=0)

    if not os.path.exists('./out/linear/' + str(N)):
        os.makedirs('./out/linear/' + str(N))
    return points


def generate_point_linear(N):
    points = np.random.rand(N, 2) * 500
    return points

def generate_centered_point(N, c):
    centeres = []
    in_center = N//c
    for i in range(0, c):
        x,y = np.random.rand(2)*500
        xs = ((np.random.rand(in_center, 1)*500)%100)+max(0, min(x - 50,500))
        ys = ((np.random.rand(in_center, 1)*500)%100)+max(0, min(y - 50,500))
        centeres.append(np.array([ [xs[i], ys[i]] for i in range(len(xs))]))
    return np.concatenate(centeres)



def visualize(snapshots, points, n, k, iters,energies, temps, k_t):
    import os
    import matplotlib.pyplot as plt
    r_path_name = k
    path_name = './out/{}/'.format(r_path_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)

    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], s=0.5)
    #plt.plot(p[:, 0], p[:, 1])
    plt.savefig(path_name +'/points{}.png'.format(n))

    for snapshot in snapshots:
        print(snapshot)

        temp, best_distance, i, solution = snapshot
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(iters, energies, 'ko-', markersize=0.02)
        plt.plot(i, best_distance, 'ro', markersize=10)
        plt.xlabel('iteracje')
        plt.ylabel('Wartość energii')

        plt.subplot(2, 1, 2)

        plt.plot(iters, temps, 'r--', markersize=0.02)
        plt.plot(i, temp, 'ro', markersize=10)
        plt.xlabel('iteracje')
        plt.ylabel('temperatura')
        plt.savefig(path_name+'Q{}Q{}Qiter'.format(k_t, n)+str(i)+'_metadata.jpg', dpi=96)
        plt.clf()
        p = np.array([points[i] for i in solution] + [points[solution[0]]])
        plt.scatter(p[:, 0], p[:, 1], s=4)
        plt.plot(p[:, 0], p[:, 1])
        plt.savefig(path_name+ 'Q{}Q{}Qiter'.format(k_t, n) + str(i) + '.jpg', dpi = 96)


    from PIL import Image
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    jpgs = list(filter(lambda x: x.startswith('Q{}Q{}Qiter'.format(k_t, n)) and  x.endswith('.jpg'), onlyfiles))
    meta = list(filter(lambda x: x.endswith('metadata.jpg'), jpgs))
    imgs = list(filter(lambda x: not x.endswith('metadata.jpg'), jpgs))
    meta = sorted(meta)
    imgs = sorted(imgs)
    imags_list = sorted([(int(imgs[i].split('iter')[1].split('.jpg')[0]),
                          'Q{}Q{}Qiter'.format(k_t,n) + (imgs[i].split('iter')[1].split('.jpg')[0]) + '_metadata.jpg', imgs[i]) for i in
                         range(len(meta))])
    import os
    if not os.path.exists('./tomp4'):
        os.makedirs('./tomp4')

    print(len(meta))


    for i in range(0, len(meta)):
        _, m, im = imags_list[i]
        print(m, im)
        imags = [Image.open(path_name+m), Image.open(path_name+im)]

        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imags])[1][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imags))

        # save that beautiful picture
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(path_name+'/finQ{}Q{}i{}.jpg'.format(k_t,n, i))




    out.write('\\item 0\\ %\n')
    out.write('\\begin {center}\n')
    out.write('\\includegraphics  {' + r_path_name + '/finQ{}Q{}i{}.jpg'.format(k_t, n, 0) + '}\\\\[1cm]\n')
    out.write('\\end {center}\n')

    out.write('\\item 40\\ %\n')
    out.write('\\begin {center}\n')
    out.write('\\includegraphics  {' + r_path_name + '/finQ{}Q{}i{}.jpg'.format(k_t, n, 2) + '}\\\\[1cm]\n')
    out.write('\\end {center}\n')

    out.write('\\item 80\\ %\n')
    out.write('\\begin {center}\n')
    out.write('\\includegraphics  {' + r_path_name + '/finQ{}Q{}i{}.jpg'.format(k_t, n, 4) + '}\\\\[1cm]\n')
    out.write('\\end {center}\n')

    out.write('\\item 100\\ %\n')
    out.write('\\begin {center}\n')
    out.write('\\includegraphics  {' + r_path_name + '/finQ{}Q{}i{}.jpg'.format(k_t, n, 5) + '}\\\\[1cm]\n')
    out.write('\\end {center}\n')





def count_distances(points, N):
    distances = np.zeros((N, N))
    for i in range(0, N-1):
        for j in range(0, N-1):
            distances[i, j] = np.linalg.norm(points[i] - points[j])

    return distances


if __name__ == "__main__":
    snapshots = []

    points_t = {'linear':lambda x: generate_point_linear(x), 'normal': lambda x: generate_point_normal(x), 'center': lambda x: generate_centered_point(x,9)}
    temps = {'-': (lambda x,y: x-y, lambda i: 0.0001,100, 0.00001, 0.001), '-f': (lambda x,y: x-y, lambda i: abs(0.00001-i*0.0000001),100, 0.0001, 0.001), 'mul': (lambda x,y: x*y, lambda i: abs(0.9999 - i*(0.0000001)),400,0.00001,0.9999999)}

    for k in points_t.keys():

        out.write('\\subsection {ROzkład'+ '{}'.format(k)+'}\n')
        for n in [10,150, 1000]:
            points = points_t[k](n)
            n = len(points)
            out.write('\\subsubsection {Rozmiar '+str(n)+'}\n')

            out.write('\\begin {center}\n')
            out.write('\\includegraphics  {' + k + '/points{}.png'.format(n) + '}\\\\[1cm]\n')
            out.write('\\end {center}\n')

            out.write('\\begin{itemize}\n')



            distances = count_distances(points,n)
            for k_t in temps.keys():
                f_t,f_i, temp,stop_temp, diff = temps[k_t]
                out.write('\\item Temperatura temp{}\n'.format(k_t))
                out.write('\\begin{itemize}\n')

                temps_val = []
                it = 0
                while temp > stop_temp:
                    it+=1
                    print(f_i(it),temp)
                    temp = f_t(temp, f_i(it))
                    temps_val.append(temp)
                iters = []
                eneregies = []
                snapshots = []
                from random import shuffle
                solution = [i for i in range(0, n)]
                shuffle(solution)
                print(solution)
                best_distance = count_distance(solution, distances)
                T = len(temps_val)
                print(T)
                for index_t in range(0, T):
                    temp = temps_val[index_t]
                    perc = index_t*100/T
                    if index_t in [1, int(0.2*T), int(0.4*T), int(0.6*T),int(0.8*T), int(0.99*T)]:
                        print('perc: {} ene : {}'.format(perc, best_distance))
                        snapshots.append((temp, best_distance, index_t, np.array(solution)))
                    i, j = swap(solution, n)
                    new_distance = count_distance(solution, distances)
                    try:
                        prop = np.math.exp((best_distance - new_distance) / max(0.004, temp))
                    except OverflowError as e:
                        prop = 0

                    if new_distance < best_distance:
                        best_distance = new_distance

                    else:
                        if prop < random.random():
                            rollback_swap(solution,i,j)
                        else:
                            best_distance = new_distance
                    iters.append(index_t)
                    eneregies.append(best_distance)


                visualize(snapshots,points,n,k,iters,eneregies, temps_val, k_t)
                out.write('\\end{itemize}\n')

            out.write('\\end{itemize}\n')



    


