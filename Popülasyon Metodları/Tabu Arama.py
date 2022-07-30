import numpy as np
import pandas as pd
import itertools as itr


## TSP (Gezgin Satıcı Problemi) için Tabu Arama algoritmasını kullanarak çözüm bulma


## Uzaklıklar matrisi (pre-determined) ##

Uzk = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

## Akış matrisi (pre-determined) ##

Aks = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])


## Baslangic Cozumu ##

x0 = ["D","A","C","B","G","E","F","H"]

# Baslangic cozumu icin dataframe olusturulması

y_uzk_df = Uzk.reindex(columns= x0 , index = x0)
y_uzk_arr = np.array(y_uzk_df)

## Baslangic cozumunun maliyeti icin dataframe olusturulması

Objfun1_basla = pd.DataFrame(y_uzk_arr*Aks)
Objfun_basla_arr = np.array(Objfun1_basla)

baslangic_toplami = sum(sum(Objfun_basla_arr))

print(baslangic_toplami)

init_final = x0[:] ## -> Sonda baslangic cozumunu yazdırmak icin kopya

print("\nBaslangic Cozumu: ",init_final)

calis = 60 ## Kac kez calistirilacak?

## Tabu Listesi olusturma ##
T_uzunluk = 10
T_list = np.empty((0,len(x0)+1))

son_cozum = []

iterasyon = 1

sonuclar = np.empty((0,len(x0)+1))

for i in range(calis):
    print("\n--> %i"% iterasyon,". iterasyon")

    # Çevresel komşuluğun oluşturulması
    N_list = list(itr.combinations(x0,2))

