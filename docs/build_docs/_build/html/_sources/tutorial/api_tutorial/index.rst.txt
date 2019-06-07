From python API
=================


Analyze Dataset
--------------------------------------------

::

  
  from sklearn import cluster, datasets, mixture
  import cluster_drug_discovery.methods.kmeans as ks

  X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
  cluster = ks.KmeansAlg(X, nclust=3)
  cluster.silhouette() 


Results:

As seen in the pictures belowe as high the silhoutte coeficient 
is the better the clustering. As the silhoutte coefficient give us
a sense of how separate each point is from the other clusters


.. figure:: images/silhouette_2.png
    :scale: 50%
    :align: center

.. figure:: images/silhouette_3.png
    :scale: 50%
    :align: center

.. figure:: images/silhouette_6.png
    :scale: 40%
    :align: center

    
