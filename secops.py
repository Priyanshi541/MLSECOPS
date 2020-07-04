
import pandas as pd

dataset = pd.read_csv('/var/www/html/logs.csv' , sep='\s+',header=None)

dataset.columns=["IP","Dash","Hyphen","Date","0000","Get","Server","Status","Length","Site"]

dataset=dataset[["IP","Date","Status"]]

dataset

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X = dataset.iloc[:,:]

x = X.to_numpy()

x

label = LabelEncoder()

IP = label.fit_transform(x[:,0])
Date = label.fit_transform(x[:,1])
Status = label.fit_transform(x[:,2])

ip = pd.DataFrame(IP, columns=['IPADD'])
date = pd.DataFrame(Date, columns=['Date'])
status = pd.DataFrame(Status, columns=['Hits'])

data = [ip, date, status]
DATASET = pd.concat(data, axis=1 )

DATASET

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_scaled = sc.fit_transform(DATASET)

import seaborn as sns

sns.set()

sns.countplot(data=DATASET , x="IPADD")

from sklearn.cluster import KMeans

model = KMeans(n_clusters=7)
model.fit(data_scaled)

pred  = model.fit_predict(data_scaled)
dataset_scaled = pd.DataFrame(data_scaled, columns=['IP', 'Date', 'Hits'])

dataset_scaled['cluster name'] = pred

IPAdd = [dataset['IP'], DATASET['IPADD']]
IPAddress = pd.concat(IPAdd, axis=1)

num = {} 
def Count(li, ip): 
    for i in li: 
        if (i in num): 
            num[i] += 1
        else: 
            num[i] = 1
    max_freq = 0
    max_key = 0
    for key, value in num.items(): 
        if value > max_freq:
            max_freq = value
            max_key = key
    return ip[li.index(max_key)], max_freq

res, freq = Count(IPAddress['IPADD'].tolist(), IPAddress['IP'].tolist())

f = open("block.txt","w")
f.write(res)
f.close()

