import random
import sys
import math
import numpy as np
import Znumbers 



class clusteringZ :

    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        


    def FindColMinMax(self,items):
        n = len(items[0])
        minima = [sys.maxsize for i in range(n)]
        maxima = [-sys.maxsize -1 for i in range(n)]

        for item in items:
            for f in range(len(item)):
                if (item[f] < minima[f]):
                    minima[f] = item[f]

                if (item[f] > maxima[f]):
                    maxima[f] = item[f]

        return minima,maxima


    def InitializeMeans(self,items, k):

        # Initialize means to random numbers between
        # the min and max of each column/feature   
        
        np.random.seed(0)
        idx = np.random.permutation(items.shape[0])[:k]
        means = items[idx]
        

        return means

    
    def TotalDistanceZ(self,x,y,w1=1/3,w2=1/3,w3=1/3):
        xzS = Znumbers.Znumbers(x[0:4],x[4:7])
        xzO = Znumbers.Znumbers(x[7:11],x[11:14])
        xzD = Znumbers.Znumbers(x[14:18],x[18:21])

        yzS = Znumbers.Znumbers(y[0:4],y[4:7])
        yzO = Znumbers.Znumbers(y[7:11],y[11:14])
        yzD = Znumbers.Znumbers(y[14:18],y[18:21])

        return w1*(Znumbers.distanceZ(xzS,yzS))+ w2*(Znumbers.distanceZ(xzO,yzO)) +w3*(Znumbers.distanceZ(xzD,yzD))


    def UpdateMean(self,items,means,belongsTo):
        
        for i in range(means.shape[0]):
            means[i] = np.mean(items[belongsTo == i], axis=0)
            #min max aggregation of Z-numbers
            # if(len(items[belongsTo==i]))==0:
            #     print("Error : the cluster ",i," contains no elements")
            #     print(means)
            #     for i in range(items.shape[0]):
            #         for j in range(means.shape[0]):
            #             print(self.TotalDistanceZ(items[i],means[j]))
            #         print("next")
            #     # for j in range(means.shape[0]): 
            #     #     print(items[belongsTo==j])
            #     break
            # for j in range(3):
            #     means[i][7*j] = np.min(items[belongsTo==i][:,7*j]) 
            #     means[i][7*j+1]=np.mean(items[belongsTo==i][:,7*j+1]) 
            #     means[i][7*j+2] = np.mean(items[belongsTo==i][:,7*j+2]) 
            #     means[i][7*j+3] = np.max(items[belongsTo==i][:,7*j+3])
            #     means[i][7*j+4]=np.min(items[belongsTo==i][:,7*j+4]) 
            #     means[i][7*j+5]=np.mean(items[belongsTo==i][:,7*j+5]) 
            #     means[i][7*j+6]=np.max(items[belongsTo==i][:,7*j+6])
            
        return means

    def Classify(self,means,item):
    
        # Classify item to the mean with minimum distance   
        minimum = np.inf
        index = -1
    
        for i in range(means.shape[0]):
    
            # Find distance from item to mean
            # print(item.shape, means[i].shape)
            dis = self.TotalDistanceZ(item, means[i])
    
            if (dis < minimum):
                minimum = dis
                index = i
        
        return index


    def CalculateMeans(self,k,items,maxIterations=1000000):
    
        # Find the minima and maxima for columns
        #cMin, cMax = self.FindColMinMax(items)
        
        # Initialize means at random points
        means = self.InitializeMeans(items,k)
        
        # Initialize clusters, the array to hold
        # the number of items in a class

    
        # An array to hold the cluster an item is in
        belongsTo = np.zeros(items.shape[0])
    
        # Calculate means
        for e in range(maxIterations):
    
            # If no change of cluster occurs, halt
            noChange = True
            for i in range(items.shape[0]):
    
                item = items[i]
    
                # Classify item into a cluster and update the
                # corresponding means.       
                index = self.Classify(means,item)
    
                
    
                # Item changed cluster
                if(index != belongsTo[i]):
                    noChange = False
    
                belongsTo[i] = index

            means = self.UpdateMean(items,means,belongsTo)
    
            # Nothing changed, return
            if (noChange):
                break
    
        return means



    def FindClusters(self,means,items):
        clusters = [[] for i in range(len(means))] # Init clusters
        labels = np.zeros(items.shape[0])
        
        for i in range(items.shape[0]):
            item = items[i]

            # Classify item into a cluster
            index = self.Classify(means,item)

            # Add item to cluster
            clusters[index].append(item)
            labels[i] = index
        
        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i])
        


        return clusters, labels

    def Fit(self,items):
        means = self.CalculateMeans(self.n_clusters,items)
        return self.FindClusters(means,items)

