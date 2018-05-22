# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:40:52 2017

@author: palom
"""
#este codigo calcula los ciclos anuales de oleaje en San Andres y hace los analisis interanuales
#Metodología: Análisis estacional
#Internaual, complementar con análisis de correlaciones
#Analisis espectral del oleaje y vientos
#>coherencia espectral
#>composite
#>periodos de retorno


 
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
import scipy.stats
import xlsxwriter as xlsxwl # Crear archivos de Excel
import pandas as pd
import scipy.stats as scp

Pathout= 'C:/Users/palom/Documents/AnalisisDatos/TrabajoFinal/'

Waves=Dataset('Oleaje.nc','r')
#print Waves.variables
#print Waves.variables.keys()
print Waves.variables['swh']
#print Waves.variables['mwd']
#print Waves.variables['mwp']
print Waves.variables['time']
tempe=Dataset('sst.mnmean.nc','r')
#tempe2=Dataset('sst.wkmean.1990-present.nc', 'r')
print tempe.variables


altura= np.array(Waves.variables['swh'][:])
direccion= np.array(Waves.variables['mwd'][:])
periodo= np.array(Waves.variables['mwp'][:])
lat= np.array(Waves.variables['latitude'][:])
lon= np.array(Waves.variables['longitude'][:])
time= np.array(Waves.variables['time'][:]).astype(np.float)

sst=np.array(tempe.variables['sst'][:])
sst_lat=np.array(tempe.variables['lat'][:])
sst_lon=np.array(tempe.variables['lon'][:])
sst_time=np.array(tempe.variables['time'][:]).astype(np.float)
#sst2=np.array(tempe2.variables['sst'][:])
#sst_lat2=np.array(tempe2.variables['lat'][:])
#sst_lon2=np.array(tempe2.variables['lon'][:])
#sst_time2=np.array(tempe2.variables['time'][:]).astype(np.float)

print tempe.variables['sst']

altura[altura==-32767]=np.nan
direccion[direccion==-32767]=np.nan
periodo[periodo==-32767]=np.nan
sst[sst==32767]=np.nan



fecha = np.array([datetime.datetime(1900,01,01)+\
datetime.timedelta(hours = time[i]) for i in range(len(time))])

#se recorta lat y lon para el punto 1

lat1=np.where(lat==12.625)[0][0]
lon1=np.where(lon==278.375)[0][0]
#se recort la info (time, lat , lon)

alt1=altura[:,lat1,lon1]
dir1=direccion[:,lat1,lon1]
per1=periodo[:,lat1,lon1]


#ciclo anual punto 1 SAI
altura1= alt1[np.isfinite(alt1)]

CicloAnual_altura1= np.zeros([12]) * np.NaN

Meses = np.array([fecha[i].month for i in range(len(fecha))])
for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
   
    altura1_tmp= altura1[tmpp]
    CicloAnual_altura1[k-1]= np.mean(altura1_tmp)

Fig= plt.figure()
plt.rcParams.update({'font.size':14})
plt.plot(CicloAnual_altura1,'-', color='skyblue',lw=3,label='Hs')
x_label = ['Año']
plt.title('Ciclo Anual Altura de Ola Significante', fontsize=24)
plt.xlabel('Mes',fontsize=18)
plt.ylabel('Hs(metros)',fontsize=18)
plt.legend(loc=0)

axes = plt.gca()
axes.set_xlim([0,11])
axes.set_ylim([0.9,2.0])
axes.set_xticks([0,1,2, 3, 4, 5, 6, 7,8, 9, 10 ,11]) #choose which x locations to have ticks
axes.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic' ]) 
plt.savefig('CicloAnualAltura1.png')

#ciclo anual con pandas

Waves=pd.Series(index=fecha, data=altura1)

WavesM=Waves.resample('M').mean()
WavesD=Waves.resample('D').mean()

WM=np.array(WavesM)
WM=WM[:-6]
WM=np.reshape(WM,(-1,12))
WMM=np.mean(WM,axis=0)
WMS=np.std(WM, axis=0)

plt.plot(WMM)



#leer txt ONI con .np

path='./nino34.txt'
Data = np.genfromtxt(path,dtype=str,skip_header=1 )
Pathout='C:/Users/palom/Documents/AnalisisDatos/TrabajoFinal/'

from datetime import datetime 

Fechas = []
Anomaly = np.zeros([len(Data), 4])
for i in range(len(Data)):
   # Fechas.append(datetime.strptime(Data[i,0], '%Y') )
    Fechas.append(datetime.strptime(Data[i,0], '%Y'))
    Anomaly[i,:] = np.array(Data[i,1].split(' ')).astype(np.int)
Fechas = np.array(Fechas)

#leer con Pandas

datos=pd.read_csv('./nino34.txt', delim_whitespace=True)
datosfecha=pd.read_csv('./nino34.txt', delim_whitespace=True,parse_dates=[[0,1]])

datos.apply(lambda x:'%s %s %s' % (x['YR'],x['MON'], x['ANOM']),axis=1)

#datos.apply(lambda x: pd.datetime.strptime("{0} {1}".format(x['YR'],\
#x['MON']), "%Y/%m"),axis=1)

fechass= datosfecha['YR_MON']

Anomaly= np.array(datosfecha['ANOM'])

Nino34= pd.DataFrame(Anomaly,index=fechass)   

#media movil
N = 3
cumsum, moving_aves = [0], []

for i, x in enumerate(Anomaly, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
        
        
DEF1950=np.mean(np.array(Anomaly[0:2]))

ONI=np.ones([811])
ONI[0]=DEF1950

for i in range(810):

    ONI[i+1]=moving_aves[i]
  
a=np.array(fechass[:-1])  

Fig= plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size':14})
plt.plot(a,ONI, 'k',label='ONI Index')
plt.axhline(y=0.5, color='r')
plt.axhline(y=-0.5,color= 'b')
axes = plt.gca()
#axes.set_xticks([1950, 1970]) #choose which x locations to have ticks
plt.savefig('ONI Index.png')

#se redondean las cifras
ONIr=[]

for i in ONI:
    ONIr.append(round(i,1))
    
ONIr=np.array(ONIr)    
#nino_nina=np.zeros([811])*np.NaN   
nino_nina=[]
    
for i in ONIr:
    if i >= 0.5:
        nino_nina.append('nino')
    elif i<= -0.5:
        nino_nina.append('nina')
    else:
        nino_nina.append(0)
    
nn=np.array(nino_nina)    
    
NNN=pd.DataFrame(nino_nina, index=a)

#determino eventos Niño, Niña y Neutros con 5 periodos consecutivos 
    
    
#fechas.append(datetime.datetime(1800,01,01)+ \

#datetime.timedelta(days = time[i]))

 
    
dates = pd.date_range('1950-01', freq='M', periods=len(ONIr))
x = ONIr
y = dates


    
NINO = y[np.all([x >0.4], axis=0)]     

NINA= y[x<-0.4]

NEUTRO= y[(x>=-0.4) & (x<=0.4) ]

tdelta=pd.Timedelta('125 days +00:00:00')
tdeltam=pd.Timedelta('32 days +00:00:00')

anosNINA=[]
for i in range(228):

    if( (NINA[i+4]-NINA[i]) < tdelta):
        anosNINA.append(NINA[i])
       
    elif((NINA[i]-NINA[i-1]<tdeltam) & (anosNINA[i-1]!='nan')) :
        anosNINA.append(NINA[i])
       
    else:
        anosNINA.append('nan')
      
        
anosNINO=[]
for i in range(225):

    if( (NINO[i+4]-NINO[i]) < tdelta):
        anosNINO.append(NINO[i])
    elif((NINO[i]-NINO[i-1]<tdeltam) & (anosNINO[i-1]!='nan')) :
        anosNINO.append(NINO[i])
    else:
        anosNINO.append('nan')
        

        
anosNINA=np.array(anosNINA)       
anosNINO=np.array(anosNINO)      
        
anosNINAS=[]

for i in anosNINA:
    if i != 'nan':
        anosNINAS.append(i)


anosNINOS=[]

for i in anosNINO:
    if i != 'nan':
        anosNINOS.append(i)
    
anosNEUTRO=[]        

for i in range(len(y)):
    
    if  np.all(((y[i] in anosNINA) ==False) & ((y[i] in anosNINO) ==False)):
        anosNEUTRO.append(y[i])
 
#FILTRO LOS DATOS DE OLEJAE PARA LOS PERIODOS NEUTRO, NINO Y NINA Y CALCULO OS CICLOS ANUALES   
    
dateWavesM=[] 
for i in range( len(WavesM)):  
    dateWavesM.append(WavesM.index[i].date())  

dateNEUTRO=[]    
for i in range( len(anosNEUTRO)):  
    dateNEUTRO.append(anosNEUTRO[i].date()) 
  
OlasNeutro=[]    

for i in range(len(dateWavesM)):
    if (dateWavesM[i] in dateNEUTRO) ==True:
        OlasNeutro.append(WavesM[i])
    else:
        OlasNeutro.append('nan')

OlasNeutro=np.array(OlasNeutro).astype(np.float)
OlasN= OlasNeutro [:-6] 
OlasNS=np.reshape(OlasN,(-1,12))
OlasNS=np.nanmean(OlasNS,axis=0)
OlasNSS=np.std(OlasNS, axis=0)    


dateNINA=[]    
for i in range( len(anosNINAS)):  
    dateNINA.append(anosNINAS[i].date()) 
  
OlasNINA=[]    

for i in range(len(dateWavesM)):
    if (dateWavesM[i] in dateNINA) ==True:
        OlasNINA.append(WavesM[i])
    else:
        OlasNINA.append('nan')

OlasNINA=np.array(OlasNINA).astype(np.float)
OlasNINA= OlasNINA [:-6] 
OlasNINAS=np.reshape(OlasNINA,(-1,12))
OlasNINAS=np.nanmean(OlasNINAS,axis=0)
OlasNINAS_S=np.std(OlasNINAS, axis=0)    

dateNINO=[]    
for i in range( len(anosNINOS)):  
    dateNINO.append(anosNINOS[i].date()) 
  
OlasNINO=[]    

for i in range(len(dateWavesM)):
    if (dateWavesM[i] in dateNINO) ==True:
        OlasNINO.append(WavesM[i])
    else:
        OlasNINO.append('nan')

OlasNINO=np.array(OlasNINO).astype(np.float)
OlasNINO= OlasNINO [:-6] 
OlasNINOS=np.reshape(OlasNINO,(-1,12))
OlasNINOS=np.nanmean(OlasNINOS,axis=0)
OlasNINOS_S=np.std(OlasNINOS, axis=0)    

  
  
plt.figure()
plt.plot(OlasN)
#plt.plot(OlasNINO, 'b')
#plt.plot(OlasNINA, 'g')  

plt.figure()
plt.plot(OlasNS)
plt.plot(OlasNINOS, 'b')
plt.plot(OlasNINAS, 'g')  

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#Correlacion entre altura de ola y temperatura en epocas neutras, nino y nina

#remuevo los valores NaN
Nino_Nnan= OlasNINO[np.isfinite(OlasNINO)]
Nina_Nnan= OlasNINA[np.isfinite(OlasNINA)]
Neutro_Nnan= OlasN[np.isfinite(OlasN)]


Corr=scp.pearsonr(Nina_Nnan, Nino_Nnan)[0]



