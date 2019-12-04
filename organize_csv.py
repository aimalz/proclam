import pylab as pl
import pandas as pd
import numpy as np

list = ['2_MikeSilogram', '3_MajorTom']

kyle = pd.read_csv('examples/plasticc/1_Kyle/1_Kyle.csv')
kylecols = kyle.columns.tolist()


matstruc = kyle.copy()
matstrucdat = np.array(matstruc)
indexlist = np.zeros(len(kylecols))
print(kylecols)
print(matstrucdat[0,:])


truth = pd.read_csv('1_Kyle/1_Kyle_truth.csv')

for file in list:
    name = file+'/'+file+'.csv'
    print(name)
    mat = pd.read_csv(name)
    matdat = np.array(mat)

    cols = mat.columns.tolist()
    index = mat.index
    #print(np.shape(mat))
    print(cols)
    print(matdat[0,:], 'test before')
    

    for i in range(len(kylecols)):
#        df[col] = df[col].replace(findL, replaceL)
        for j in range(len(cols)):
            if cols[j]==kylecols[i]:
                indexlist[i]=j
                matstrucdat[:,i] = matdat[:,j]
                print(i, j, cols[j], kylecols[i])

                matstruc.rename(columns={"A": "a", "B": "c"})
    print(matstrucdat[0,:], 'test after')

    newname = file+'/'+file+'_reordered.csv'
    newnametruth = file+'/'+file+'_reordered_truth.csv'

    matstruc.to_csv(newname, index=False)
    truth.to_csv(newnametruth)
