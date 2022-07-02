from numpy import exp
from numpy import sin
from numpy import tan
from numpy import e
from numpy import pi
from numpy import sqrt
from numpy import cos
from numpy import meshgrid
from numpy import asarray
from numpy import arange
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import math

def amac_ciz(x,y):
    return sin(x) + tan(y) + pow(1.25,x+y)
def amac_fonksiyonu(v):
       x,y = v
       return  sin(x) + tan(y) + pow(1.25,x+y)

def sinir_kontrol(nokta,sinirlar):
    for d in range(len(sinirlar)):
        if nokta[d] < sinirlar[d,0] or nokta[d] > sinirlar[d,1]:
            return False
    return True

def tepeTirmanma(objective,sinirlar,iterasyon_siniri,delta):
    cozum = None
    cozumler = []
    girdiler = []
    while cozum is None or not sinir_kontrol(cozum,sinirlar):
        cozum = sinirlar[:,0] + rand(len(sinirlar)) * (sinirlar[:,1] - sinirlar[:,0])
    mevcut_cozum = objective(cozum)
    for i in range(iterasyon_siniri):
        #ilk adim
        aday_cozum = None
        while aday_cozum is None or not sinir_kontrol(aday_cozum,sinirlar):
            aday_cozum = cozum + randn(len(sinirlar)) * delta
        aday_cozumu_ata = objective(aday_cozum)
        if aday_cozumu_ata >= mevcut_cozum:
            cozum, mevcut_cozum = aday_cozum,aday_cozumu_ata
            girdiler.append(aday_cozum)
            cozumler.append(mevcut_cozum)
            print('>%d f(%s) = %.5f' % (i,cozum,mevcut_cozum))


    return [cozum,mevcut_cozum,cozumler,girdiler]

#pseudorandom sayı oluşturucu seed'i
seed(1)
sinirlar = asarray([[0.0,10.0],[0.0,10.0]])
iterasyon_siniri = 1000
delta = 0.1
eniyi, skor, cozumler,girdiler = tepeTirmanma(objective,sinirlar,iterasyon_siniri,delta)
print('Done!')
print('f(%s) = %f' % (eniyi,skor))
print('\nBulunan cozumler {}'.format(cozumler))
print('\nGirdiler ve bulunan amaç fonksiyonu değerleri: \n')
i = 0
for element in girdiler:

    for nested_tem in element:
        print(nested_tem , end=" ")
    print(cozumler[i])
    print(cozumler[i-1]-cozumler[i])
    i=i+1

    print()


#Amaç fonksiyonun 3 boyutlu grafiğini yazdırma

x_ = arange(0.0,10.0,delta)
y_ = arange(0.0,10.0,delta)
x,y = meshgrid(x_,y_)
results = amac_ciz(x,y)
figure = pyplot.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(x,y,results,cmap='jet')
pyplot.show()

#

# Tanım aralığında rasthele bir nokta oluşturma








