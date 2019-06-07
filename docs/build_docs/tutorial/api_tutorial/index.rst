From python API
=================


Analyze Dataset
--------------------------------------------

Tha analysis is based on:

- Silhouette index:  The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Therefore, as high as possible the better.

- Calinski-Harabasz index: The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.

- David-Bouldin index: The score refers to a model with better separation between the clusters.

::

  
  from sklearn import cluster, datasets, mixture
  import cluster_drug_discovery.methods.kmeans as ks

  X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
  cluster = ks.KmeansAlg(X, nclust=3)
  cluster.analysis_run() 


Silhouette
````````````````````
As seen in the pictures belowe as high the silhoutte coeficient 
is the better the clustering. As the silhoutte coefficient give us
a sense of how separate each point is from the other clusters.

Therefore, when we move from 3 to 4 clusters the silhouette index
of the first and second cluster clearly drops as they are next to each 
other in the feature space while with nclusters=3 all clusters were very
separated.


.. figure:: images/silhouette_2.png
    :scale: 50%
    :align: center

.. figure:: images/silhouette_3.png
    :scale: 50%
    :align: center

.. figure:: images/silhouette_4.png
    :scale: 50%
    :align: center

Carlinski
``````````````````

In terms of the  Calinski-Harabasz index the score is higher when clusters are dense and well separated,
which relates to a standard concept of a cluster. The score is fast to compute.
 Whereas, The Calinski-Harabasz index is generally higher for convex clusters than other concepts of clusters,
such as density based clusters like those obtained through DBSCAN.

As seen in the plot belowe as Calinski index is based only on density it gives a better 
score for ncluster=4 than ncluster=3 as the clusters are smaller and therefore more dense.
However, that does not mean that the result is better. In conclusion, this index give us usufull
information about the cluster density but must be use with other indexes like the silhoutte.


.. figure:: images/calinski_score.png
    :scale: 80%
    :align: center


David-Bouldin
```````````````````````

The David-Bouldin score refers to a model with better separation between the clusters since algorithms that produce clusters with low intra-cluster distances (high intra-cluster similarity) and high inter-cluster distances (low inter-cluster similarity) will have a low Davies–Bouldin index, the clustering algorithm that produces a collection of clusters with the smallest Davies–Bouldin index is considered the best algorithm based on this criterion.

As seen in the plot below we can clearly differenciate that moving from nclust=2 to nclust=3 the intra-cluster
distance gets better. However, from nclust=3 and nclust=4 the intra-cluster distance remains the same and the 
values does no change that much.


.. figure:: images/davidbouldin_score.png
    :scale: 80%
    :align: center


Analysis Conclusion
`````````````````````

All indexes give useful information, then the most efficient anlysis
to set the clustering parameters would be to put the three of them
together with some visualization tool (dimensionallity reduction).
