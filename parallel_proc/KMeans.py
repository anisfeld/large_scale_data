import pandas as pd
import numpy as np
import os
import time
import math
import multiprocessing as mp
from ggplot import *


def parallel_proc_k_means(data, k, nprocs=3, sleep = .035, random = True, TIMER = False, TIMER_2=True):
    '''
    Find k-means using a mp.Queue and mp.Processes

    input: 
        data (np.array or pd.dataFrame) should only include continuous data
        k (int) number of means
        nprocs (int) number of processors
        sleep (float) wait time to ensure queue is in order (sleep )
        random (Boolean) whether to use a random set of starting means or the top k rows
        TIMER (Boolean) determines whether or not we get detailed timing information
    '''
    
    data = np.array(data)
    centroids = init_centroids(k, data, random)
    convergence_met = False
    
    j = 0
    if TIMER or TIMER_2:
        print("parallel_proc_{}_means with {} processors".format(k,nprocs))
        start_time = time.time()
        lap_time = start_time

   
    while not convergence_met:
        assignment_q = mp.Queue()
        chunksize = int(math.ceil(len(data) / float(nprocs)))
        
        procs = []

        for i in range(nprocs):

            p = mp.Process(target = points_to_centroids,
                        args = (data[chunksize * i:chunksize * (i + 1)], 
                                centroids, 
                                assignment_q))
            p.start()

            procs.append(p)
            
            time.sleep(sleep)
        
        
        assignments = np.array([])
        for i in range(nprocs):
            assignments = np.r_[assignments, assignment_q.get()]


        new_centroids = recenter(assignments, data, k)

        for p in procs:
            p.join()


        convergence_met = criteria(centroids, new_centroids)
        centroids = new_centroids
        
        assignment_q.close()
        assignment_q.join_thread()
        
        j += 1
        if TIMER:
            current_time = time.time()
            print("Iteration {} took  {}".format(j, current_time - lap_time))
            lap_time = current_time

    
    if TIMER or TIMER_2:
        total = time.time() - start_time
        print("{} iterations took {:.3f} seconds ({:.3f} per iteration).".format(j, total, total/j))

    return centroids, assignments


def pool_k_means(data, k, nprocs=3, chunksize=300, random = True, TIMER = False, TIMER_2=True):
    '''
    Find k-means using a mp.pool

    input: 
        data (np.array or pd.dataFrame) should only include continuous data
        k (int) number of means
        nprocs (int) number of processors
        chunksize (int) number of rows sent to each worker for a given process
        random (Boolean) whether to use a random set of starting means or the top k rows
        TIMER (Boolean) determines whether or not we get detailed timing information
    '''
    global centroids

    data = np.array(data)
    centroids = init_centroids(k, data, random)
    convergence_met = False
    
    j = 0
    if TIMER or TIMER_2:
        print("pool_{}_means with {} processors".format(k, nprocs))
        start_time = time.time()
        lap_time = start_time

    
    while not convergence_met:
        
        pool = mp.Pool(nprocs)
        
        # Using pool.apply_async was very slow mostly because the result
        # is stored as a mp.object that requires a .get() call as far as I can tell.
        # TASKS = [(data[i], centroids) for i in range(data.shape[0])]
        # assignments =  [pool.apply_async(p2c, t) for t in TASKS]
        # assignments = np.array([a.get() for a in assignments])

        #import pdb; pdb.set_trace()
        assignments = np.array([i for i in pool.imap(p2c, data, chunksize=chunksize)])
        pool.terminate()
        pool.join()
        
        new_centroids = recenter(assignments, data, k)
        convergence_met = criteria(centroids, new_centroids)
        centroids = new_centroids
        
        j += 1
        if TIMER:
            current_time = time.time()
            print("Iteration {} took  {}".format(j, current_time - lap_time))
            lap_time = current_time

    
    if TIMER or TIMER_2:
        total = time.time() - start_time
        print("{} iterations took {:.3f} seconds ({:.3f} per iteration).".format(j, total, total/j))
    
    return centroids, assignments



def k_means(data, k, random = True, TIMER=False, TIMER_2=True):
    '''
    Find k-means without parallel processing

    input: 
        data (np.array or pd.dataFrame) should only include continuous data
        k (int) number of means
        random (Boolean) whether to use a random set of starting means or the top k rows
        TIMER (Boolean) determines whether or not we get detailed timing information
    '''
    
    data = np.array(data)
    centroids = init_centroids(k, data, random)
    convergence_met = False
    
    j = 0
    if TIMER or TIMER_2:
        print("single processor {}_means".format(k))
        start_time = time.time()
        lap_time = start_time    


    while not convergence_met:  
        assignments = points_to_centroids(data, centroids)
        new_centroids = recenter(assignments, data, k)
        convergence_met = criteria(centroids, new_centroids)
        centroids = new_centroids

        j += 1
        if TIMER:
            current_time = time.time()
            print("Iteration {} took  {}".format(j, current_time - lap_time))
            lap_time = current_time

    
    if TIMER or TIMER_2:
        total = time.time() - start_time
        print("{} iterations took {:.3f} seconds ({:.3f} per iteration).".format(j, total, total/j))
       
    return centroids, assignments
    

def init_centroids(k, data, random = True):
    '''
    Initialize centroids
    Inputs:
        k (int)
        data (np.array)
    
    returns k centroids (np.array)
    ''' 
    if random:
        # shuffle introduces a bug if there's no copy.
        data = data.copy()
        np.random.shuffle(data)
    
    return data[0:k]

def distance(v1, v2):
    '''
    return L2 norm distance 
    '''
    if len(v1.shape)==1:
        v1 = v1[np.newaxis,:]

    return np.sqrt(np.sum((v1 - v2)**2, axis=1))



def point_to_centroids(point, centroids):
    '''
    assigns point to the nearest centroid

    input: point (np.array of shape (d,))
           centroids (np.array of shape (k,d))

    returns: int between 0 and k-1
    '''
    return np.argmin(distance(centroids, point))

def p2c(point):
    '''
    wrapper function for mp.pool.map; centroids is a global
    '''
    return point_to_centroids(point, centroids)

   
def points_to_centroids(points, centroids, out_q= None):
    '''
    assigns all points to centroids

    input: point (np.array of shape (N,d))
           centroids (np.array of shape (k,d))

    returns: np.array of length N

    '''
    # centroids include assignment column
    N = points.shape[0]
    assignments = np.zeros(N)
    
    for i in range(N):
        assignments[i] = point_to_centroids(points[i], centroids)
    
    if out_q is not None:
        out_q.put(assignments)
    else:
        return assignments


# 3. 
def recenter(assignments, points, k):
    '''
    Take mean per cluster

    input: assignments (np.array of length N)
           points (np.array of shape (N,d))
           k (int)
    
    returns: np.array of shape (k,d) with updated centroids
    '''
    centroids = np.zeros((k, points.shape[1] ))

    for i in range(k):
        centroids[i] = np.mean(points[assignments == i], axis=0)

    return centroids


def criteria(centroids, new_centroids, min_delta = 1):
    '''
    Test stopping criteria
    min_delta = 0 is equivalent to no new assignments
    '''
    return np.mean((centroids - new_centroids)**2) < min_delta
    

def kmeans2way(d):
    ''' Make scatterplot of points in 2-d space with group assignments

    Inputs: d (pd.DataFrame) three columns 
            column 0: 'groups'
            column 1 & 2: data
    '''

    #d.loc['group'] = int(d.loc['group'])
    return ggplot(d, aes(x=d.columns[1], y=d.columns[2], color=d.columns[0])) + geom_point()