MOwNiT2 - lab2
++++++++++++++

Zadanie 1
=========


Metoda Gausa Jordana


.. code-block:: python

    def gauss_jordan(A,b):
        n = len(A)
        #make one matrix
        for i in range(0,n):
            A[i].append(b[i])


        for i in range(0, n):

            pivot, pivot_row = abs(A[i][i]), i
            for k in range(i+1, n):
                if abs(A[k][i]) > pivot:
                    pivot,pivot_row = abs(A[k][i]),k

            #swap
            for k in range(i, n+1):
                tmp = A[pivot_row][k]
                A[pivot_row][k] = A[i][k]
                A[i][k] = tmp

            #elimnation
            for k in range(i+1, n):
                elimination_ratio = -A[k][i]/A[i][i]
                for j in range(i, n+1):
                    if i == j:
                        A[k][j] = 0
                    else:
                        A[k][j] += elimination_ratio * A[i][j]

        #back substition
        x = [0 for i in range(n)]
        for i in range(n-1, -1, -1):
            x[i] = A[i][n]/A[i][i]
            for k in range(i-1, -1, -1):
                A[k][n] -= A[k][i] * x[i]
        return x


Zadanie 2
=========
Faktoryzacja LU



.. code-block:: python


    import numpy as npimport numpy as np
    def lu(a):
        n = len(a)

        L = np.zeros((n, n))
        for i in range(0, n):
            L[i, i] = 1

        U = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                U[i, j] = a[i, j]

        for i in range(0, n):
            max_index = i
            pivot = a[i, i]
            for k in range(i,n):
                if abs(a[k, i]) > abs(pivot) :
                    max_index, pivot = k, a[k, i]

            #swap
            if max_index != i:
                a[i, max_index] = a[max_index, i]

            #elimination
            for k in range(i + 1, n):
                elimination_ratio = U[k, i] / (U[i, i])
                L[k, i] = elimination_ratio
                for j in range(i, n):
                    U[k, j] -= elimination_ratio * U[i, j]

            for k in range(i + 1, n):
                U[k, i] = 0

        return L, U



Zadanie 3
=========


Dla rozwiązania układów elektrycznych korzystam z pakietu networkx


Rozwiązanie za pomocą praw Kirchoffa
------------------------------------


1) Dla każdego wierzchołka buduje mapę wierzchołków wejściowych i wyjściowych
2) Dla mapy wierzchołków wyjściowych i wejściowych generuję równania z I prawa Kirchoffa
3) Szukam cykli
4) Dla każdego cyklu generuje równania z II prawa Kirchoffa
5) Rozwiązuję macierz



.. code-block:: python

    import networkx as nx
    import numpy as np

    class KirchoffCirucitResolver():
        def __init__(self, lines, start, end, power_volate):
            self.lines = lines
            self.start = start
            self.end = end
            self.power_volate = power_volate
            self.weight_map = {}
            self.edges_map = {}
            self.node_inp_map = {}
            self.node_out_map = {}

        def resolve(self):
            self.find_cycles()
            self.display_cycles()
            self.build_node_inp_out_map()
            self.print_outs_and_inps()
            self.init_intense_map_to_index()
            self.first_law()
            self.second_law()
            self.solve_matrix()


        def find_cycles(self):
            self.g = nx.Graph()
            edge_to_weihgt_map = {}
            lines.append('{} {} {}'.format(self.start, 'x', 0))
            lines.append('{} {} {}'.format(self.end, 'x', 0))
            for line in self.lines:
                a, b, v = tuple(line.split())
                self.weight_map[(a, b)] = v
                self.weight_map[(b, a)] = v
                self.g.add_edge(a, b)

            cycles = list(nx.cycle_basis(self.g))
            self.cycles = cycles

        def display_cycles(self):
            print('Found Cycles')
            for cycle in self.cycles:
                print(cycle)

        def build_node_inp_out_map(self):
            self.init_node_inp_out_map()
            for cycle in self.cycles:
                i = 0
                while i < len(cycle):
                    current = cycle[i]
                    output = cycle[(i + 1) % len(cycle)]
                    input = cycle[(i - 1) % len(cycle)]
                    if 'x' in cycle:
                        if current == 'x' or input == 'x':
                            self.node_inp_map[current].append(input)
                        if current == 'x' or output == 'x':
                            self.node_out_map[current].append(output)

                    else:
                        self.node_out_map[current].append(output)
                        self.node_inp_map[current].append(input)

                    i += 1

        def init_node_inp_out_map(self):
            for node in self.g.nodes:
                self.node_inp_map[node] = []
                self.node_out_map[node] = []

        def print_outs_and_inps(self):
            for node in self.g.nodes:
                print('node {} input {} output {}'.format(node, self.node_inp_map[node], self.node_out_map[node]))

        def firs_law(self):
            pass

        def init_intense_map_to_index(self):
            i = 0
            self.intense_index = {}
            self.intense_index_rev = {}

            for (a, b) in self.g.edges:
                self.intense_index[(a, b)] = i
                self.intense_index[(b, a)] = i
                self.intense_index_rev[i] = (a, b)
                i += 1
            print('Intense maping')
            for k in self.intense_index.keys():
                print('edge {} : {}'.format(k, self.intense_index[k]))

        def first_law(self):
            self.A = []
            self.b = []
            print('First law, generated:')
            s = ''
            for i in range(0, len(self.g.edges)):
                s += '{}|'.format(self.intense_index_rev[i])
            print(s)

            for node in self.g.nodes:
                eq = np.zeros(len(self.g.edges))

                for outp in self.node_out_map[node]:
                    eq[self.intense_index[node, outp]] = -1

                for inp in self.node_inp_map[node]:
                    eq[self.intense_index[inp, node]] = 1

                print('node {}, eq: {}'.format(node, eq))
                self.A.append(eq)
                self.b.append(0)

        def second_law(self):

            for cycle in self.cycles:
                eq = np.zeros(len(self.g.edges))
                i = 0
                r = 0
                while i < len(cycle):
                    fr, to = cycle[i], cycle[(i + 1) % len(cycle)]
                    eq[self.intense_index[(fr, to)]] = self.weight_map[(fr, to)]

                    if 'x' in cycle:
                        r = -self.power_volate
                    i += 1

                print('cycle {}, eq{} = {}'.format(cycle, eq, r))
                self.A.append(eq)
                self.b.append(r)

            pass

        def solve_matrix(self):
            a = np.array(self.A)
            b = np.array(self.b)

            AT = a.transpose()
            A = np.dot(AT, a)
            Y = np.dot(AT, b)

            from scipy.linalg import solve

            x = solve(A, Y)

            print(x)
            pass


    def draw_graph(g, node_from, node_to):
        graph = nx.DiGraph()
        labels = {}

        for k in g.keys():
            f, t = k
            graph.add_edge(f, t, weight=g[k], label=g[k])
            labels[(f, t)] = g[k]

        intense = sum(map(lambda x: g[x], graph.edges(node_from)))

        weights = [graph[u][v]['weight'] for u, v in graph.edges]

        pos = nx.circular_layout(graph)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(graph, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=weights, arrows=True)

        # labels
        nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        import matplotlib.pyplot as plt

        plt.axis('off')
        plt.show()

        nx.draw(g, nx.circular_layout(g), edge_labels=labels, edges=g.edges, width=weights)



Metoda potencjałow wezłowych
----------------------------


0) Szukam węzłów (wierzchołki o stopniu conajmniej 3)
1) Dla każdego znalezionego węzła szukam sąsiednie węzły
2) Dla każdego węzła generuję jego równanie uwzględniając konduktancje do sąsiednich węzłów
3) Rozwiązuje macierz
4) Dla otrzymanych potencjałów liczę napięcia na gałęziach
5) Z prawa Ohma obliczam natężenia na gałęziach


.. code-block:: python

    import networkx as nx
    import numpy as np


    class NodalCircuitResovler():
        def __init__(self, lines, pow_from, pow_target, pow_voltage):
            self.pow_target = pow_target
            self.pow_voltage = pow_voltage
            self.pow_from = pow_from
            self.lines = lines
            self.weight_of_edge = {}

        def resolve(self):
            self.generate_cycles()
            self.print_cycles()
            self.find_nodes()
            self.print_nodes()
            self.find_neighbours_of_nodes()
            self.print_neighbours()
            self.build_matrix()
            self.display_matrix()

            self.solve_sys_eq()
            self.display_nodes_voltage()

            return self.build_edge_list_with_intenses()

        def generate_cycles(self):
            self.g = nx.Graph()

            for line in self.lines:
                a, b, v = tuple(line.split())
                self.weight_of_edge[(a, b)] = float(v)
                self.weight_of_edge[(b, a)] = float(v)
                self.g.add_edge(a, b)
            self.g.add_edge(self.pow_from, self.pow_target)
            self.weight_of_edge[(self.pow_from, self.pow_target)] = 0.0
            self.weight_of_edge[(self.pow_target, self.pow_from)] = 0.0

            self.cycles = list(nx.cycle_basis(self.g))

        def print_cycles(self):
            print('Found cycles')
            for cycle in self.cycles:
                print(cycle)

        def find_nodes(self):
            self.nodes = []
            for n in self.g.nodes:
                if len(list(self.g.neighbors(n))) > 2:
                    self.nodes.append(n)

        def print_nodes(self):
            print('Found nodes')
            for node in self.nodes:
                print(node)

        def find_neighbours_of_nodes(self):
            self.neighbour_map = {}
            for node in self.nodes:
                self.neighbour_map[node] = []

                for cycle in self.cycles:
                    if node in cycle:
                        nodes_in_cycle = list(filter(lambda x: x in self.nodes, cycle))
                        index_of_node = nodes_in_cycle.index(node)
                        len_n = len(nodes_in_cycle)
                        len_c = len(cycle)
                        before = nodes_in_cycle[(index_of_node - 1) % len_n]
                        after = nodes_in_cycle[(index_of_node + 1) % len_n]
                        before_index = cycle.index(before)
                        after_index = cycle.index(after)
                        index_of_node = cycle.index(node)
                        i = index_of_node
                        b_res = [node]
                        while i != before_index:
                            i = (i - 1) % len_c
                            b_res.append(cycle[i])
                        i = index_of_node
                        a_res = [node]
                        while i != after_index:
                            i = (i + 1) % len_c
                            a_res.append(cycle[i])
                        if a_res not in self.neighbour_map[node]:
                            self.neighbour_map[node].append(a_res)
                        if b_res not in self.neighbour_map[node]:
                            self.neighbour_map[node].append(b_res)

        def print_neighbours(self):
            for k in self.neighbour_map.keys():
                print('{} has neighbours: {}'.format(k, self.neighbour_map[k]))
            pass

        def build_matrix(self):

            A = []
            b = []
            n = len(self.neighbour_map)

            self.node_to_column_map = {}
            actual_index = 0

            # init node to col map
            for k in self.neighbour_map.keys():
                self.node_to_column_map[k] = actual_index
                actual_index += 1

            # make a eq for one node
            for k in self.neighbour_map.keys():
                if k == self.pow_from:
                    eq = np.zeros(n)
                    eq[self.node_to_column_map[k]] = 1
                    b.append(0)
                    A.append(eq)
                elif k == self.pow_target:
                    eq = np.zeros(n)
                    eq[self.node_to_column_map[k]] = 1
                    b.append(power)
                    A.append(eq)
                else:
                    eq = self.count_conductance(self.neighbour_map[k])
                    b.append(0)
                    A.append(eq)
            self.A = A
            self.B = b

            pass

        def count_conductance(self, paths):
            eq = np.zeros(len(self.nodes))

            # soource
            source_val = 0.0
            for path in paths:
                path_conductance = self.count_path_conductance(path)
                if path_conductance != 0:
                    source_val += path_conductance

            eq[self.node_to_column_map[paths[0][0]]] = source_val

            # neighbours
            #
            counductances_map = self.build_counductances_map(paths)

            for path in paths:
                s = path[0]
                t = path[-1]
                # print('count n')
                target = path[-1]
                eq[self.node_to_column_map[target]] = -counductances_map[(s, t)]

            return eq

        def count_path_conductance(self, path):
            i = 1
            n = len(path)
            resistance = 0.0

            while i < n:
                resistance += self.weight_of_edge[(path[i - 1], path[i])]
                i += 1
            if resistance == 0:
                return 0

            return 1 / resistance

        def build_counductances_map(self, paths):
            counductances = {}

            for path in paths:
                source = path[0]
                target = path[-1]
                counductances[(source, target)] = []

            for path in paths:
                source = path[0]
                target = path[-1]
                cond = 0.0
                i = 1
                while i < len(path):
                    cond += self.weight_of_edge[(path[i - 1], path[i])]
                    i += 1
                counductances[(source, target)].append(cond)

            for k in counductances.keys():
                if len(counductances[k]) > 1:
                    cval = 0.0
                    for edge in counductances[k]:
                        cval += 1 / edge
                    counductances[k] = cval
                else:
                    counductances[k] = 1 / counductances[k][0]

            return counductances

        def display_matrix(self):
            inv_map = {v: k for k, v in self.node_to_column_map.items()}
            s = ''
            for k in sorted(inv_map.keys()):
                s += '\t{}'.format(inv_map[k])
            print("----------------")
            print(s)
            for a in self.A:
                print(a)
            pass

        def solve_sys_eq(self):
            inv_map = {v: k for k, v in self.node_to_column_map.items()}
            x = np.linalg.solve(np.array(self.A), np.array(self.B))
            res = {}
            for i in range(len(x)):
                res[inv_map[i]] = x[i]
            self.nodes_voltage = res

        def display_nodes_voltage(self):
            print('Nodes voltage')
            for node in self.nodes_voltage.keys():
                print('Vnode{} = {}'.format(node, self.nodes_voltage[node]))

        def build_edge_list_with_intenses(self):
            edges = {}

            for k in self.neighbour_map.keys():

                for path in self.neighbour_map[k]:

                    source, target = path[0], path[-1]
                    u = abs(self.nodes_voltage[source] - self.nodes_voltage[target])
                    i = 1
                    res = self.count_path_conductance(path)
                    if res != 0:
                        res = 1 / res
                    while i < len(path):
                        if res != 0:
                            edges[(path[i - 1], path[i])] = u / res
                        i += 1

            return edges



Przykład
--------
Graf losowy, z napięciem 20V między węzłami 1 a 3

.. image:: https://raw.githubusercontent.com/moskalap/mownit-lab/master/lab/lab2/res/img/losowy.png

Graf 2D - niestety problem z wizualizacją

.. image:: https://raw.githubusercontent.com/moskalap/mownit-lab/master/lab/lab2/res/img/2d-cut.png

Graf kubiczny

.. image:: https://raw.githubusercontent.com/moskalap/mownit-lab/master/lab/lab2/res/img/kubiczny-cut.png







