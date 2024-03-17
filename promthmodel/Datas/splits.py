import pandas as pd 
from sklearn.model_selection import train_test_split

dt = pd.read_csv('data.csv')
dt = dt.drop(columns=['Unnamed: 0'])
train , test = train_test_split(dt , test_size=0.5)
train.to_csv('promthmodel/Datas/Train.csv' , index = False)
test.to_csv('promthmodel/Datas/Test.csv', index = False)

print(train)
