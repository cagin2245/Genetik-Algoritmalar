import numpy as np
from numpy import exp
from numpy import sin
from numpy import tan
from numpy import e
from numpy import pi
from numpy import sqrt
from numpy import cos
import random as rd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import time


def amac_fonksiyonu(x, y):
    return sin(x) + tan(y) + pow(1.25, x + y)


def GenetikAlgoritma(kromozom):
    lb_x, ub_x = 0, 10
    len_x = (len(kromozom) // 2)
    lb_y, ub_y = 0, 10
    len_y = (len(kromozom) // 2)

    hassasiyet_x = (ub_x - lb_x) / ((2 ** len_x) - 1)
    hassasiyet_y = (ub_x - lb_y) / ((2 ** len_y) - 1)

    z = 0
    t = 1
    x_bit_sum = 0
    for i in range(len(kromozom) // 2):
        x_bit = kromozom[-t] * (2 ** z)
        x_bit_sum += x_bit
        t = t + 1
        z = z + 1

    z = 0
    t = 1 + (len(kromozom) // 2)
    y_bit_sum = 0
    for j in range(len(kromozom) // 2):
        y_bit = kromozom[-t] * (2 ** z)
        y_bit_sum = y_bit_sum + y_bit
        t = t + 1
        z = z + 1

    decoded_x = (x_bit_sum * hassasiyet_x) + lb_x
    decoded_y = (y_bit_sum * hassasiyet_y) + lb_y
    amac_fonksiyonu_degeri = amac_fonksiyonu(decoded_x, decoded_y)
    return decoded_x, decoded_y, amac_fonksiyonu_degeri


def parents_ts(all_solutions):
    parents = np.empty((0, np.size(all_solutions, 1)))
    for i in range(2):
        index_list = np.random.choice(len(all_solutions), 3, replace=False)
        olasi_ebeveyn1 = all_solutions[index_list[0]]
        olasi_ebeveyn2 = all_solutions[index_list[1]]
        olasi_ebeveyn3 = all_solutions[index_list[2]]

        obj_parent1 = GenetikAlgoritma(olasi_ebeveyn1)[2]
        obj_parent2 = GenetikAlgoritma(olasi_ebeveyn2)[2]
        obj_parent3 = GenetikAlgoritma(olasi_ebeveyn3)[2]

        min_obj_func = min(obj_parent1, obj_parent2, obj_parent3)
        if min_obj_func == obj_parent1:
            secilen_ebeveyn = olasi_ebeveyn1
        elif min_obj_func == obj_parent2:
            secilen_ebeveyn = olasi_ebeveyn2
        else:
            secilen_ebeveyn = olasi_ebeveyn3

        parents = np.vstack((parents, secilen_ebeveyn))
    parent_1 = parents[0, :]
    parent_2 = parents[1, :]
    return parent_1, parent_2


def crossover(parent_1, parent_2, prob_crossover=0.01):
    child_1 = np.empty((0, len(parent_1)))
    child_2 = np.empty((0, len(parent_2)))

    if np.random.rand() < prob_crossover:

        index_1 = np.random.randint(0, 1)
        index_2 = np.random.randint(0, 1)

        while index_1 == index_2:
            index_2 = np.random.randint(0, len(parent_2))
        if index_1 < index_2:
            first_seg_par_1 = parent_1[:index_1]
            mid_seg_par_1 = parent_1[index_1:index_2 + 1]
            last_seg_par_1 = parent_1[index_2 + 1:]
            first_seg_par_2 = parent_2[:index_1]
            mid_seg_par_2 = parent_2[index_1:index_2 + 1]
            last_seg_par_2 = parent_2[index_2 + 1:]
            child_1 = np.concatenate((first_seg_par_1, mid_seg_par_1, last_seg_par_1))
            child_2 = np.concatenate((first_seg_par_2, mid_seg_par_2, last_seg_par_2))
        else:
            first_seg_par_1 = parent_1[:index_2]
            mid_seg_par_1 = parent_1[index_2:index_1 + 1]
            last_seg_par_1 = parent_1[index_1 + 1:]
            first_seg_par_2 = parent_2[:index_2]
            mid_seg_par_2 = parent_2[index_2:index_1 + 1]
            last_seg_par_2 = parent_2[index_1 + 1:]
    else:
        child_1 = parent_1
        child_2 = parent_2
    return child_1, child_2


def mutation(child_1, child_2, mutation=0.1):
    mutated_child_1 = np.empty((0, len(child_1)))
    ix = 0  # child_1 ilk indexi
    for i in child_1:
        mutateBool = np.random.rand()

        if mutateBool < mutation:
            if child_1[ix] == 0:
                child_1[ix] = 1
            else:
                child_1[ix] = 0
            mutated_child_1 = child_1
            ix = ix + 1
        else:
            mutated_child_1 = child_1
            ix = ix + 1

    mutated_child_2 = np.empty((0, len(child_2)))
    ix = 0  # child_2 ilk indexi
    for i in child_2:
        mutateBool = np.random.rand()

        if mutateBool < mutation:
            if child_2[ix] == 0:
                child_2[ix] = 1
            else:
                child_2[ix] = 0
            mutated_child_2 = child_2
            ix = ix + 1
        else:
            mutated_child_2 = child_2
            ix = ix + 1
    return mutated_child_1, mutated_child_2


if __name__ == '__main__':
    kromozom = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                         0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0])

    population = 200
    prob_mutation = 0.3
    generation = 180
    crossover_prob = 1
    all_solutions = np.empty((0, len(kromozom)))
    best_of_gen = np.empty((0, len(kromozom) + 1))
    for i in range(population):
        rd.shuffle(kromozom)
        all_solutions = np.vstack((all_solutions, kromozom))
    strt_time = time.time()
    gen = 1
    for j in range(generation):
        new_pop = np.empty((0, len(kromozom)))
        new_pop_obj = np.empty((0, len(kromozom) + 1))
        sorted_pop_obj = np.empty((0, len(kromozom) + 1))

        print("--> Jenerasyon: #", gen)

        aile = 1
        for k in range(int(population / 2)):
            print("--> Aile: #", aile)
            parent_1 = parents_ts(all_solutions)[0]
            parent_2 = parents_ts(all_solutions)[1]

            child_1 = crossover(parent_1, parent_2, prob_crossover=crossover_prob)[0]
            child_2 = crossover(parent_1, parent_2, prob_crossover=crossover_prob)[1]

            m_child_1 = mutation(child_1, child_2, mutation=prob_mutation)[0]
            m_child_2 = mutation(child_1, child_2, mutation=prob_mutation)[1]

            obj_m_child_1 = GenetikAlgoritma(m_child_1)[2]
            obj_m_child_2 = GenetikAlgoritma(m_child_2)[2]

            m1_obj = np.hstack((obj_m_child_1, m_child_1))
            m2_obj = np.hstack((obj_m_child_2, m_child_2))

            new_pop = np.vstack((new_pop, m_child_1, m_child_2))
            new_pop_obj = np.vstack((new_pop_obj, m1_obj, m2_obj))
            aile = aile + 1
        all_solutions = new_pop
        sorted_pop_obj = np.array(
            sorted(new_pop_obj, key=lambda x: x[0]))  # 0'ıncı index (amaç fonksiyonu değeri) değerine göre sıralıyoruz
        best_of_gen = np.vstack((best_of_gen, sorted_pop_obj[
            0]))  # jenerasyondaki en iyi amaç fonksiyonu değeri ilk eleman olmalı, sorted_pop_obj fonksiyonun index[0] değeri
        gen = gen + 1
    end_time = time.time()
    sorted_last_pop = np.array(sorted(new_pop_obj, key=lambda x: x[0]))
    sorted_best_gen = np.array(sorted(best_of_gen, key=lambda x: x[0]))

    best_str_convergence = sorted_last_pop[0]
    best_str_all = sorted_best_gen[0]

    print("exec time: ", end_time - strt_time)
    print()
    print("Son çözüm: ", best_str_convergence[1:])
    print("Kodlanmış X (convr.): ", best_str_convergence[1:14])
    print("Kodlanmış X (convr.): ", best_str_convergence[14:])
    print("Son çözüm:  ", best_str_all[1:])
    print("Kodlanmış X (en iyi): ", best_str_all[1:14])
    print("Kodlanmış Y (en iyi): ", best_str_all[14:])

    final_cozum_convergence = GenetikAlgoritma((best_str_convergence[1:]))
    final_cozum_all = GenetikAlgoritma(best_str_all[1:])

    print()
    print("Decode X (conv): ", round(final_cozum_convergence[0], 2))
    print("Decode Y (conv): ", round(final_cozum_convergence[1], 2))
    print("OBJ (conv): ", round(final_cozum_convergence[2], 2))
    print()
    print("Decode X (Best): ", round(final_cozum_all[0], 2))
    print("Decode Y (Best): ", round(final_cozum_all[1], 2))
    print("OBJ (Best): ", round(final_cozum_all[2], 2))
    best_obj_val_conv = (best_str_convergence[0])
    best_obj_val_all = best_str_all[0]

    # Amaç fonksiyonun 3 boyutlu grafiğini yazdırma
'''
    x_ = np.arange(-6.0, 6.0,0.1)
    y_ = np.arange(-6.0, 6.0,0.1)
    x, y = np.meshgrid(x_, y_)
    results = amac_fonksiyonu(x, y)
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    pyplot.show()
'''
