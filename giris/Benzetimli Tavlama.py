from numpy import mgrid
from numpy import asarray
from numpy import arange
from numpy import sin
from numpy import tan
from numpy import exp
from numpy import meshgrid
from numpy.random import rand
from numpy.random import randn
from numpy.random import seed
import numpy.random as nprd
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d

def objective_draw(x,y):
    return sin(x) + tan(y) + pow(1.25,x+y)

def objective(v):
    x,y = v
    return sin(x) + tan(y) + pow(1.25,x+y)



def benzetimliTavlama(objective,sinirlar,iterasyon_siniri,delta,isi):
    best = sinirlar[:,0] + rand(len(sinirlar)) * (sinirlar[:,1]- sinirlar[:,0])
    best_eval = objective(best)
    mev, mev_eval = best, best_eval
    skorlar = list()
    #algoritma
    for i in range(iterasyon_siniri):
        candidate = mev + randn(len(sinirlar)) * delta
        candidate_eval = objective(candidate)
        if candidate_eval > best_eval:
            best, best_eval = candidate,candidate_eval
            skorlar.append(best_eval)
            print('>%d f(%s) = %.5f' % (i,best,best_eval))
        diff = candidate_eval - mev_eval
        t = isi / float(i+1)
        metropolis = exp(-diff / t)

        #yeni noktayı tutup tutmayacağımızın kontrolü
        if diff < 0 or rand()< metropolis:
            mev,mev_eval = candidate,candidate_eval
    return[best,best_eval,skorlar]

seed(1)
sinirlar = asarray([[0.0,10.0],[0.0,10.0]])
iterasyon_siniri = 1000
delta = 0.1
isi = 100

best, skor, tümSkorlar = benzetimliTavlama(objective,sinirlar,iterasyon_siniri,delta,isi)
print('f(%s) = %f' % (best,skor))
pyplot.plot(tümSkorlar, '.-')
pyplot.xlabel('Gelişim numarası')
pyplot.ylabel('Eval f(x)')
pyplot.get_current_fig_manager().set_window_title('Amaç Fonksiyonu Maksimizasyonu')
pyplot.show()


'''
sinirlar = asarray([[0.0,10.0],[0.0,10.0]])
girdiler = mgrid[0.0:10.0:0.1, 0.10:10.0:0.1].reshape(2,-1).T
sonuclar = [objective(x,y)for x,y in girdiler]
x_optima = 0.0
x_ = arange(0.0,10.0,0.1)
y_ = arange(0.0,10.0,0.1)
x,y = meshgrid(x_,y_)
sonuclard = objective_draw(x,y)
figure = pyplot.figure(1)
axis = figure.gca(projection = '3d',)
axis.plot_surface(x,y,sonuclard,cmap = 'jet')

pyplot.get_current_fig_manager().set_window_title('Amaç Fonksiyonu')

#pyplot.show()


iterations = 100
initial_temp = 10

iterations =[i for i in range(iterations)]
temperatures = [initial_temp/float(i+1) for i in iterations]

#farklı deltalar

differences = [0.01,0.1,1.0]
f3 = pyplot.figure(3)
for d in differences:
    metropolis = [exp(-d/t) for t in temperatures]
    # iterasyonlar vs metropolis
    label = 'diff=%0.2f'% d
    pyplot.plot(iterations,metropolis,label=label)
pyplot.xlabel('İterasyon')
pyplot.ylabel('Metropolis Criterion')
pyplot.legend()
pyplot.get_current_fig_manager().set_window_title('Isı vs Metropolis')



#pyplot iterations vs temperatures
f1 = pyplot.figure(2)
plt2 = pyplot.plot(iterations, temperatures)
pyplot.xlabel('Iteration')
pyplot.ylabel('Temperature')
pyplot.get_current_fig_manager().set_window_title('Isı vs İterasyon')
pyplot.show()
'''


