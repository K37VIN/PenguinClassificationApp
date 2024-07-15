import pandas as pd  
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("C:/Users/KIIT0001/Documents/kaggle_datasets/penguins_cleaned.csv")

## target : 'species'
## encode :['sex','island']

# creating dummy variables for the columns to be encoded 

new_sex=pd.get_dummies(df['sex'],prefix='sex')
new_island=pd.get_dummies(df['island'],prefix='island')

df.columns=df.columns.astype(str)
df=pd.concat([df['species'], df['bill_length_mm'], df['bill_depth_mm'],
       df['flipper_length_mm'], df['body_mass_g'],new_sex,new_island],axis=1)

#target encoding
target_map={
    'Adelie':0,
    'Chinstrap':1,
    'Gentoo':2
}
def target_encode(val):
    return target_map[val]

df['species']=df['species'].apply(target_encode)

#split x and y
y=df['species']
x=df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
       'body_mass_g', 'sex_female', 'sex_male', 'island_Biscoe',
       'island_Dream', 'island_Torgersen']]

# model training
clf=RandomForestClassifier()
clf.fit(x,y)

#model pickling
import pickle as pkl
pkl.dump(clf,open('penguin_classification.pkl','wb'))