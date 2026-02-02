import h5py
import numpy as np
import pandas as pd

def ReadDat(file_name, NMems, Nref) : 
    '''
    Reads data stored in a .dat file. The data must have been recorded by NMems Mems and NMems analog channels
    '''
    data = np.fromfile(file_name, dtype='int32')
    Sigs = np.reshape(data,(-1, NMems+Nref+1)).T #V
    Count = Sigs[0,:]
    Mems = Sigs[1:NMems+1,:]
    Ref =  Sigs[-Nref:,:]
    Ntime = len(Count)
    return Ntime, Mems, Ref


separator = ':'     #Charactère qui sépare l'indice de début et de fin dans le tableur
data_type = 'i'     #Type des données pour les signaux temporels, automatiquement changé en complex pour les siggnaux fréquentiels
Nmems = 256
fs = 50e3

ref_file = 'C:/Augustin/TN10/violons/SignalPos.xlsx'    #Nom du fichier de référence avec les indices temporels

# format = 'time'                                       #Créer la base de données en temporel ou fréquentiel
format = 'freq'

filename = f'C:/Augustin/TN10/violons/Database_{format}.hdf5'   #Nom du fichier de la base de données




with h5py.File(filename,'w') as f :
    dic = pd.read_excel('C:/Augustin/TN10/violons/SignalPos.xlsx', sheet_name = None)

    for subfold in dic.keys() :
        dataframe = dic[subfold]        
        if subfold == 'hammer' :
            source = 'marteau'
            Nref = 3
        else : 
            source = 'Victor'
            Nref = 2

        for row in dataframe.index :
            fold = dataframe.at[row, dataframe.columns[0]]


            signal_file = 'C:/Augustin/TN10/violons/ManipViolon_Anech_08062023/' + source + '/' + fold + '/' + subfold + '.dat'
            Nsamples,Mems,Refs = ReadDat(signal_file, NMems = Nmems, Nref=  Nref)
            

            for ind_repet in dataframe.columns[1:] :
                Poskey = f'{subfold}_{fold}_{ind_repet}'
                print('writing ' + Poskey)
                Pos = np.array(dataframe.at[row,ind_repet].split(separator), dtype=float)
                print(Pos)
                data = Mems[:,int(Pos[0]*fs):int(Pos[1]*fs)]
                if source == 'marteau' : 
                    ref = Refs[:,int(Pos[0]*fs):int(Pos[1]*fs)]
                    if data.shape[1] > 10000 :
                        data = data[:10000]
                        ref = ref[:10000]
                    elif data.shape[1] < 10000 :
                        data = np.pad(data,((0,0),(0,1)), constant_values=0)
                        ref = np.pad(ref, ((0,0),(0,1)), constant_values=0)
                else : ref = Refs[0,int(Pos[0]*fs):int(Pos[1]*fs)]

                

                if format == 'freq' : 
                    data = np.fft.rfft(data)
                    ref = np.fft.rfft(ref)
                    data_type = 'complex'

                data_set = f.create_dataset(f'{fold}/{subfold}/Measure_{ind_repet}', data.shape, dtype = data_type)
                data_set[...] = data

                ref_set = f.create_dataset(f'{fold}/{subfold}/Ref_{ind_repet}', ref.shape, dtype = data_type)
                ref_set[...] = ref
