from math import dist
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

    N_sayac = 0

    i_komsu = np.empty((0,len(x0)))

    for i in N_list:
      x_swap = []
      a_sayac = N_list[N_sayac]
      A_1 = a_sayac[0] # ilk eleman
      A_2 = a_sayac[1] # ikinci eleman

      # Yeni olusturulan bölümlerden iki seçenek için yeni liste oluşturma
      u = 0
      for j in x0:
        if x0[u] == A_1:
          x_swap = np.append(x_swap,A_2)
        elif x0[u] == A_2:
          x_swap = np.append(x_swap,A_1)
        else:
          x_swap = np.append(x_swap, x0[u])
        
        x_swap = x_swap[np.newaxis] # Swap işleminden sonra yeni x0 oluşturulması
        
        u = u+1
      

      i_komsu = np.vstack((i_komsu, x_swap)) # tüm komsu cözümlerin swap islemi yapilmis hali
      N_sayac = N_sayac +1
  


    OF_i_N = np.empty((0,len(x0)+1)) # Amaç fonksiyonu değerini dahil etmek için +1 büyüklükte boş dizi
    OF_N = np.empty((0,len(x0)+1)) 
    
    N_sayac = 1
    # N. komşudaki i. çözümün amaç fonksiyonunun hesaplanması
    for i in i_komsu:
      y_uzk_df = Uzk.reindex(columns = i, index=i)
      y_uzk_arr = np.array(y_uzk_df)

      # Başlangıç çözümünün maliyeti için dataframe oluşturulması
      Objfun1_basla = pd.DataFrame(y_uzk_arr*Aks)
      Objfun_basla_arr = np.array(Objfun1_basla)

      i_komsu_maliyet = sum(sum(Objfun_basla_arr))
      i = i[np.newaxis]

      OF_i_N = np.column_stack((i_komsu_maliyet,i)) # Amaç fonksiyonu değerlerinin seçilen parça vektörü (Ör: B,C gibi) üzerine işlenmesi (Stack)
      OF_N = np.vstack((OF_N, OF_i_N))
      N_sayac = N_sayac + 1
    # Komşuların sıralı Amaç fonksiyonu değerleri
    Of_N_sorted = np.array(sorted(OF_N,key=lambda x: x[0]))

    
    # Çözüm Tabu Listesinde yer alıyor mu kontrol et, yer alıyorsa sonrakini seç
    t = 0
    guncel_cozum = Of_N_sorted[t] # Güncel çözüm
    
    while guncel_cozum[0] in T_list[:,0]:
      guncel_cozum = Of_N_sorted[t]
      t = t+1
    
    if len(T_list) >= T_uzunluk: # Tabu listesi doluysa
      T_list = np.delete(T_list,(T_uzunluk-1),axis=0) # Son satırı sil
    T_list = np.vstack((guncel_cozum,T_list))

    sonuclar = np.vstack((guncel_cozum,sonuclar)) # Her çalıştırmada elde edilen en iyi sonucu kaydet

    # Yerel minimumda sıkıştığında, aramayı tekrar başlatmak için. (Diversification)

    Mod_Iter = iterasyon % 10
    
    Rnd_1 = np.random.randint(1,len(x0)+1)
    Rnd_2 = np.random.randint(1,len(x0)+1)
    Rnd_3 = np.random.randint(1,len(x0)+1)

    if Mod_Iter == 0:
      Xt = []
      A1 = guncel_cozum[Rnd_1]
      A2 = guncel_cozum[Rnd_2]

      # Yeni kümeyle yeni liste oluşturma
      s_temp = guncel_cozum

      w = 0
      for i in s_temp:
        if s_temp[w] == A1:
          Xt = np.append(Xt, A2)
        elif s_temp[w] == A2:
          Xt = np.append(Xt, A1)
        else:
          Xt = np.append(Xt, s_temp[w])
        w = w+1
      guncel_cozum = Xt

      # Aynı olan kümeler değiştiriliyor
      Xt = []
      A1 = guncel_cozum[Rnd_1]
      A2 = guncel_cozum[Rnd_2]

      # Yeni kümeyle yeni liste oluşturma
      w = 0
      for i in guncel_cozum:
        if guncel_cozum[w] == A1:
          Xt = np.append(Xt,A2)
        elif guncel_cozum[w] == A2:
          Xt = np.append(Xt,A1)
        else:
          Xt = np.append(Xt,guncel_cozum[w])
        w = w + 1
      guncel_cozum = Xt
    X0 = guncel_cozum[1:]
    iterasyon = iterasyon + 1

    # Her 5 çalışmanın ardından Tabu Listesi'nin uzunluğu 5 ile 20 arasında değiştir
    if Mod_Iter == 5 or Mod_Iter == 0 : 
      T_uzunluk = np.random.randint(5,20)
  
t= 0
son_c = []
for i in sonuclar:

  if(sonuclar[t,0])<= min(sonuclar[:,0]):
    son_c = sonuclar[t,:]
  t = t +1

son_cozum = son_c[np.newaxis]

print("\n\nDinamik Tabu Listesi")
print()
print("Başlangıç Çözümü:",init_final)
print("Başlangıç maliyeti",baslangic_toplami)
print()
print("Tüm iterasyonlardaki en küçük değer: ",son_cozum)
print("En düşük maliyet: ",son_cozum[:,0])

