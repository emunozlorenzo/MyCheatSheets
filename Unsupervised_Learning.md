<p align="center"> 
<img src="https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/image2.png">
</p>

# Unsupervised Learning

![alt text](https://github.com/emunozlorenzo/MasterDataScience/blob/master/img/icon2.png "Logo Title Text 1") [Eduardo Muñoz](https://www.linkedin.com/in/eduardo-mu%C3%B1oz-lorenzo-14144a144/)

![Determining the number of clusters in a data set](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)

## 1. Distance Matrix

*Normalizar las variables antes de aplicar la matriz de distancias*

```python
# Import Libraries
from scipy.spatial import distance_matrix # To calculate the ditance_matrix
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D # To show a 3d plot

# Import Dataset
data = pd.read_csv('../datasets/movies/movies.csv',sep=';')
data
```
|    user_id |   star_wars |   lord_of_the_rings |   harry_potter |
| ----------|------------|--------------------|---------------|
|          1 |         1.2 |                 4.9 |            2.1 |
|          2 |         2.1 |                 8.1 |            7.9 |
|          3 |         7.4 |                 3   |            9.9 |
|          4 |         5.6 |                 0.5 |            1.8 |
|          5 |         1.5 |                 8.3 |            2.6 |
|          6 |         2.5 |                 3.7 |            6.5 |
|          7 |         2   |                 8.2 |            8.5 |
|          8 |         1.8 |                 9.3 |            4.5 |
|          9 |         2.6 |                 1.7 |            3.1 |
|         10 |         1.5 |                 4.7 |            2.3 |

```python
movies = data.columns.values.tolist()[1:] # List of column names: star_wars,lord_of_the_rings,harry_potter

dm1 = distance_matrix(data[movies],data[movies],p=1) # Manhattan Distance
dm2 = distance_matrix(data[movies],data[movies],p=2) # Euclidean Distance
dm10 = distance_matrix(data[movies],data[movies],p=10) # p>>> distance<<<

# Function to convert distance_matrix to a Dataframe
def distance_matrix_to_df(dd,col_name):
    return pd.DataFrame(dd,index=col_name,columns=col_name)
    
distance_matrix_to_df(dm1,data['user_id'])
distance_matrix_to_df(dm2,data['user_id'])
distance_matrix_to_df(dm10,data['user_id'])
```
Example: Distance Matrix (Manhattan Distance)

|     1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |    9 |   10 |
| -----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
|   0   |  9.9 | 15.9 |  9.1 |  4.2 |  6.9 | 10.5 |  7.4 |  5.6 |  0.7 |
|   9.9 |  0   | 12.4 | 17.2 |  6.1 |  6.2 |  0.8 |  4.9 | 11.7 |  9.6 |
|  15.9 | 12.4 |  0   | 12.4 | 18.5 |  9   | 12   | 17.3 | 12.9 | 15.2 |
|   9.1 | 17.2 | 12.4 |  0   | 12.7 | 11   | 18   | 15.3 |  5.5 |  8.8 |
|   4.2 |  6.1 | 18.5 | 12.7 |  0   |  9.5 |  6.5 |  3.2 |  8.2 |  3.9 |
|   6.9 |  6.2 |  9   | 11   |  9.5 |  0   |  7   |  8.3 |  5.5 |  6.2 |
|  10.5 |  0.8 | 12   | 18   |  6.5 |  7   |  0   |  5.3 | 12.5 | 10.2 |
|   7.4 |  4.9 | 17.3 | 15.3 |  3.2 |  8.3 |  5.3 |  0   |  9.8 |  7.1 |
|   5.6 | 11.7 | 12.9 |  5.5 |  8.2 |  5.5 | 12.5 |  9.8 |  0   |  4.9 |
|   0.7 |  9.6 | 15.2 |  8.8 |  3.9 |  6.2 | 10.2 |  7.1 |  4.9 |  0   |

```python
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xs=data['star_wars'],ys=data['lord_of_the_rings'],zs=data['harry_potter']);
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/3dplot.png">
</p>

## 2. Hierarchical Clustering

### 2.1 Linkage Criterions

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/linkages.png">
</p>

#### 2.1.1 Single (Enlace Simple)
- Uses the __minimum__ of the distances between all observations of the two clusters
- La distancia entre dos clusters es el __mínimo__ de las distancias entre cualquier dos puntos del cluster 1 y el cluster 2 

#### 2.1.2 Complete (Enlace Completo)
- Uses the __maximum__ of the distances between all observations of the two clusters
- La distancia entre dos clusters es el __máximo__ de las distancias entre cualquier dos puntos del cluster 1 y el cluster 2 

#### 2.1.3 Average (Enlace Promedio)
- Uses the __average__ of the distances between all observations of the two clusters
- La distancia entre dos clusters es el __promedio__ de las distancias entre cualquier dos puntos del cluster 1 y el cluster 2 

#### 2.1.4 Centroid (Enlace del Centroide)
- Distances between __centroids__ of two clusters
- La distancia entre dos clusters es la distancia entre el __centroide__ (punto medio) del cluster 1 y el del cluster 2 

#### 2.1.5 Ward (Enlace de Ward)
- Minimizes the variance of the clustes bieng merged
- Los clusters minimizan la varianza dentro de los puntos del mismo y en el dataset global 

```python
# Hierarchical Clustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage: WARD Distance: EUCLIDEAN 
Z = linkage(data[movies],method='ward',metric='euclidean') # data[movies] definido arriba

# Plot Dendrogram 
plt.figure(figsize=(25,10))
plt.title('Dendograma para el Clustering Jerarquico')
plt.xlabel('ID usuarios Netflix')
plt.ylabel('Distancia')
dendrogram(Z, leaf_rotation=0, leaf_font_size=10)
plt.show();
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/ward.png">
</p>

```python
# Linkage: CENTROID Distance: EUCLIDEAN 
Z = linkage(data[movies],method='centroid',metric='euclidean') # data[movies] definido arriba

# Plot Dendrogram 
plt.figure(figsize=(25,10))
plt.title('Dendograma para el Clustering Jerarquico')
plt.xlabel('ID usuarios Netflix')
plt.ylabel('Distancia')
dendrogram(Z, leaf_rotation=0, leaf_font_size=10)
plt.show();
```

<p align="center"> 
<img src="https://github.com/emunozlorenzo/MyCheatSheets/blob/master/img/centroid.png">
</p>

#### Example: Hierarchical Clustering 
- X dataset (array n x m) de puntos a clusterizar
- n número de datos (Rows)
- m número de rasgos (Columns)
- Z array de enlace del cluster con la info de uniones
- k número de clusters

![**Notebook: Example Hierarchical Clustering using Python**](https://github.com/emunozlorenzo/MachineLearning/blob/master/09_Clustering/05_Clustering_Jerarquico_Completo_Perfect.ipynb)

## 3. KMeans
