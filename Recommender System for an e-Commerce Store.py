#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
import seaborn as sb


# In[2]:


#load datatsets

decdf = pd.read_csv ('/Users/admin/Desktop/eCommerce Events History in Cosmetics Shop/2019-Dec.csv')
novdf = pd.read_csv ('/Users/admin/Desktop/eCommerce Events History in Cosmetics Shop/2019-Nov.csv')
octdf = pd.read_csv ('/Users/admin/Desktop/eCommerce Events History in Cosmetics Shop/2019-Oct.csv')
jandf = pd.read_csv ('/Users/admin/Desktop/eCommerce Events History in Cosmetics Shop/2020-Jan.csv')
febdf = pd.read_csv ('/Users/admin/Desktop/eCommerce Events History in Cosmetics Shop/2020-Feb.csv')


# In[3]:


#concate the datasets for 5 month period into 1 df

ecommdf = pd.concat([decdf, novdf, octdf, jandf,febdf])


# In[4]:



#Outlier detection

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


# In[5]:


#Due to the inability to use the lage volume of datre, we will be redusing the dimension (columns) of the dataframe.
ecommdfb = ecommdf.drop(['user_id','category_id','category_code','user_session' ], axis =1)


# In[6]:


ecommdfb.columns =['event_time','event_type','product_id','brand','price']
data = ecommdfb.iloc[:, 0:4]
target =ecommdfb.iloc[:,4]
ecommdfb [:4]


# In[7]:




sb.boxplot(x = 'brand',y='price', data = ecommdfb, palette = 'hls')


# In[8]:


#The volume of the dataset is too large to use box hence from below we cansee that the minimum pricew is -7.937000e+01​.
#prices are not expected to be negative.
#Describe ecommdf
ecommdf.describe()


# In[ ]:





# In[9]:


ecommdf.head()


# In[10]:


#Display attributes and datatypes
ecommdf.info()


# In[11]:


#DEscribe the data
ecommdf.describe()


# In[12]:


ecommdf.describe(include = np.object)


# In[13]:


#show details of users interacions(events) on the site
ecommdf['event_type'].value_counts () 


# In[14]:


#show details of users interacions(events) on the site for each month
from IPython.display import display
display(decdf['event_type'].value_counts ())
display(novdf['event_type'].value_counts ()) 
display(octdf['event_type'].value_counts ())
display(jandf['event_type'].value_counts ())
display(febdf['event_type'].value_counts ())



# In[ ]:





# In[15]:



Decevent_type = ['view','cart','remove_from_cart','purchase']
value_counts = [498166, 283752, 204009, 62648]

fig, decdf = plt.subplots()
decdf.pie(value_counts, labels = Decevent_type, autopct = '%1.1f%%')             
decdf.axis('equal')
decdf.set_title('Event Type')


# In[16]:



Decevent_type = ['view','cart','remove_from_cart','purchase']
value_counts = [498166, 283752, 204009, 62648]

fig, decdf = plt.subplots()
decdf.pie(value_counts, labels = Decevent_type, autopct = '%1.1f%%')             
decdf.axis('equal')
decdf.set_title('Event Type for December 2019')


Novevent_type = ['view','cart','remove_from_cart','purchase']
value_counts = [2076132, 1311807, 925481, 322417]

fig, novdf = plt.subplots()
novdf.pie(value_counts, labels = Novevent_type, autopct = '%1.1f%%')             
novdf.axis('equal')
novdf.set_title('Event Type for November 2019')


Octevent_type = ['view','cart','remove_from_cart','purchase']
value_counts = [1862164, 1232385, 762110, 245624]

fig, octdf = plt.subplots()
octdf.pie(value_counts, labels = Octevent_type, autopct = '%1.1f%%') 
octdf.axis('equal')
octdf.set_title('Event Type for October 2019')


Janevent_type = ['view','cart','remove_from_cart','purchase']
value_counts = [2037608, 1148323, 815024, 263797]

fig, jandf = plt.subplots()
jandf.pie(value_counts, labels = Janevent_type, autopct = '%1.1f%%')             
jandf.axis('equal')
jandf.set_title('Event Type for January 2020')




event_type = ['view','cart','remove_from_cart','purchase']
value_counts = [1953586, 1148694, 812409, 241993]

fig, febdf = plt.subplots()
febdf.pie(value_counts, labels = event_type, autopct = '%1.1f%%')             
febdf.axis('equal')
febdf.set_title('Event Type for February 2020')


# In[17]:


#Most popular products among users in descending order
ecommdf ['product_id'].value_counts()


# In[18]:


#Top 10 Popular brands for the 5-month period
Popular_brands = ecommdf['brand'].value_counts()
Popular_brands.head(10)


# In[19]:


Popular_brands = ['runail','irisk','masura','grattol','ingarden','bpw.style','estel','kapous','uno','jessnail']
value_counts = ['1348995', '928932', '788203', '740615','387742','383395','303627','271074','221589','216006']

fig, popular_brands = plt.subplots()
popular_brands.pie(value_counts, labels = Popular_brands, autopct = '%1.1f%%')             
popular_brands.axis('equal')
popular_brands.set_title('Top Ten Brands')


# In[20]:


#products purchased by brand name
Products_purchased = ecommdf.loc[ecommdf.event_type == 'purchase']
Products_purchased


# In[21]:


#Top 50 Users (Customers)
Top_customers = Products_purchased.groupby('user_id').user_id.agg([len]).sort_values(by= 'len', ascending = False)
Top_customers.head(50)


# In[22]:


#Best time users make purchases( to target them with recommendations and adverts)
Best_sales_time = Products_purchased.groupby('event_time').event_time.agg([len]).sort_values(by= 'len', ascending = False)
Best_sales_time.head(20)


# In[23]:


#length of ecommrdf 
len(ecommdf)


# In[24]:


#Number of unique users(customers)
users = ecommdf['user_id'].unique()
len(users)


# In[25]:



#Number of unique products
products = ecommdf['product_id'].unique()
len(products)


# In[26]:


#Number of unique products
products = ecommdf['product_id'].unique()
len(products)


# In[27]:


ecommdf.corr()


# In[28]:



## Recommender System using  method = pearson
ecommdfR = ecommdf.drop(['event_time','category_id','category_code','user_session' ], axis =1)



# In[29]:


#since prices cannot be begative, we exclude all prices less than 1
ecommdfR= ecommdfR[ecommdfR.price >= 1]


# In[30]:




## To build the recommender system event_type will be ranked in order of interest and stored as ratings in a new column.
ecommdfR ['rating'] = ecommdfR['event_type'].map({'remove_from_cart' :1, 'view': 2, 'cart': 3, 'purchase':4})
ecommdfR.head()


# In[31]:


ecommdfRS = ecommdfR.drop(['event_type' ], axis =1)
ecommdfRS.head()


# In[32]:


#mean of rating
Mean_of_rating = ecommdfRS['rating'].mean()
Mean_of_rating


# In[33]:


#To drop duplicated records
ecommdfRS = ecommdfRS.drop_duplicates()


# In[34]:


ecommdfRSP =ecommdfRS.pivot_table(index = 'user_id', columns = 'brand', values = 'rating')
ecommdfRSP.head(20)


# In[35]:


X = ecommdfRSP['runail']


# In[36]:


corr = ecommdfRSP.corrwith(X)


# In[37]:


matrix_corr = ecommdfRSP.corr(method = 'pearson', min_periods = 100)
matrix_corr.head()


# In[38]:


user_corr = pd.Series()


# In[39]:


ecommdfRSP.iloc[0].dropna()


# In[40]:


user_id = 0


# In[41]:



for brand in ecommdfRSP.iloc[user_id].dropna().index:
    corr_list = matrix_corr[brand].dropna()*ecommdfRSP.iloc[user_id][brand]


# In[42]:


user_corr = user_corr.append(corr_list)


# In[43]:


brand_list =[]
for i in range(len(ecommdfRSP.iloc[user_id].dropna().index)):
    if ecommdfRSP.iloc[user_id].dropna().index[i] in user_corr:
        brand_list.append(ecommdfRSP.iloc[user_id].dropna().index[i])
    else:
        pass
    user_corr = user_corr.drop(brand_list)


# In[44]:



print('Hi, based on the brand that you have bought: \n')
for i in ecommdfRSP.iloc[user_id].dropna().index:
    print(i)
print( '\n you would definitely love these brands \n')
for i in user_corr.sort_values(ascending = False). index[:5]:
    print(i)


# In[45]:


## Recommender System using KNN, metric = cosine

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# In[46]:


pip install fuzzywuzzy


# In[47]:



from fuzzywuzzy import process, fuzz


# In[48]:


#Spare Matrix

#df too large when index= product_id, therefore we use brand instead

ecommdf_KNN3 = ecommdfRS.pivot_table(index = 'brand', columns = 'user_id', values = 'rating').fillna(0)


# In[49]:



mat_ecommdf_KNN3 = csr_matrix(ecommdf_KNN3.values)


# In[50]:



#Euclidean Distance
#Manhattan Distance
#Minkowski Distance
#Cosine Similarity. Calculates similariries btw ... e.g.(Cos)
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 30)


# In[51]:


model_knn.fit(mat_ecommdf_KNN3)


# In[53]:


#Recommender (brand_name) => List of brands recommended

def recommender (brand_name, data,model, n_recommendations):
    model.fit(data)
    idx = process.extractOne(brand_name ,ecommdfRS['brand'])[2]
    print('brands Selected: ', ecommdfRS['brand'][idx], 'Index: ',idx)
    print('Searching for recommendations...')
    distances, indices = model.kneighbors(data[idx], n_neighbors = n_recommendations)
    print(distances, indices)
    
recommender('runail', mat_ecommdf_KNN3, model_knn, 30)


# In[ ]:


#Due to memomy contrains in weka, we reduce the number of months from 5 to 2 namely Dec 2019,November 2019 and Oct 2019.


# In[ ]:


#explore and preprocess df for weka

ecommdf1 = ecommdf.drop(['event_time', 'category_id','category_code','user_session' ], axis =1)


# In[ ]:


#drop event_type
ecommdfW= ecommdf2.drop(['event_type' ], axis =1)


# In[ ]:


#save on directory for weka
ecommdfW.to_csv('ecommdfW.csv', index = False)


# In[ ]:


ecommdf3m = pd.concat([decdf, novdf, octdf])
ecommdf3m.head()


# In[ ]:


ecommdf13m = ecommdf3m.drop(['event_time','user_id','category_id','category_code','user_session' ], axis =1)


# In[ ]:


## To build the recommender system event_type will be ranked in order of interest and stored as ratings in a new column.
ecommdf13m ['rating'] = ecommdf3m['event_type'].map({'remove_from_cart' :1, 'view': 2, 'cart': 3, 'purchase':4})
ecommdf13m.head()


# In[ ]:


ecommdf23m = ecommdf13m.drop(['event_type' ], axis =1)
ecommdf23m.head()


# In[ ]:


#since prices cannot be begative, we exclude all prices less than 1
ecommdf23m= ecommdf13m[ecommdf13m.price >= 1]


# In[ ]:


#drop event_type
ecommdfW3m= ecommdf23m.drop(['event_type' ], axis =1)


# In[ ]:


#save on directory for weka
ecommdfW3m.to_csv('ecommdfW3m.csv', index = False)


# In[ ]:




