from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics as ms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



def analysis_run(X, cluster_obj, range_n_clusters = [2, 3, 4, 5, 6]):

    silhouette_scores = []
    calisnki_scores = []
    david_bouldin_scores = []

    if hasattr(cluster_obj, "nclust"):
        range_n_clusters = range_n_clusters
        output_pattern = "silhouette_nclust{}.png"
    elif hasattr(cluster_obj, "epsilon"):
        range_n_clusters = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
        output_pattern = "silhouette_epsilon{}.png"
    
    for n_clusters in range_n_clusters:
        output = output_pattern.format(n_clusters)
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if hasattr(cluster_obj, "nclust"):
            cluster_obj.nclust = n_clusters
        elif hasattr(cluster_obj, "epsilon"):
            cluster_obj.epsilon = n_clusters


        cluster_labels = cluster_obj.run()
        n_clusters = len(list(set(cluster_labels)))
        calinski_score = ms.calinski_harabaz_score(X, cluster_labels)
        calisnki_scores.append(calinski_score)
        david_bouldin_score = davies_bouldin_score(X, cluster_labels)
        david_bouldin_scores.append(david_bouldin_score)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg,
              "The david bouldin score is:", david_bouldin_score,
              "The calinski_score is:", calinski_score)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        try:
            centers = cluster_obj._clusterer.cluster_centers_
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
    
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
    
        except AttributeError:
            pass
        # Draw white circles at cluster centers
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.savefig(output)
    silhouette_avg = silhouette_score(X, cluster_labels)
    #Calinski Score
    fig, ax = plt.subplots(1, 1)
    ax.plot(range_n_clusters, silhouette_scores, marker="o")
    ax.set_title("Silhouette index score vs number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette index score")
    #Calinski Score
    fig, ax = plt.subplots(1, 1)
    ax.plot(range_n_clusters, calisnki_scores, marker="o")
    ax.set_title("Calinski Score vs number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Calinski index score")
    plt.savefig("silhouette_score.png")
    #David Bouldin index
    fig, ax = plt.subplots(1, 1)
    ax.plot(range_n_clusters, david_bouldin_scores, marker="o")
    ax.set_title("David Bouldin Score vs number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("David Bouldin index score")
    plt.savefig("davidbouldin_score.png")