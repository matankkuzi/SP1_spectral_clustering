/*
 ============================================================================
 Name        : kmeans.c
 Description : This file contains the K-means algorithm,
                supporting functions,and the c-python api.
 ============================================================================
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "matoperations.h"

/* c_kmeans variables. In general scope for memory management */
double *a_clustersum;
int *a_clusters;
mat *input_mat, *centroid_mat, *tempcentroid_mat;


/* step 2a of the kmeans algorithm:
  Assign each observation xi to cluster Sj,
  that minimizes the distance between xi to the the centroid.*/
static void assign_clusters(mat *input_mat, mat *centroid_mat, int *a_clusters, int k, int n, int d){
    int i, j, min_distance_centroid;
    double min_distance, temp_distance;

    /* iterate over all data points */
    for(i=0; i<n; i++){

        /* initialize the minimal distance according to the first centroid */
        min_distance = distance(get_row(input_mat, i),get_row(centroid_mat, 0),d);
        min_distance_centroid = 0;

        /* iterate over all centroids to get the true minimal distance */
        for(j=0; j<k; j++){
            temp_distance = distance(get_row(input_mat, i),get_row(centroid_mat, j),d);
            if(temp_distance < min_distance){
                min_distance = temp_distance;
                min_distance_centroid = j;
            }
        }

        /* assign the cluster with regards to the minimal distance */
        a_clusters[i] = min_distance_centroid;
    }
}

/* a c function to perform step 2 of kmeans */
static int *c_kmeans(double *p, int *kClusters, int k, int n, int d, int MAX_ITER ){
    /* initialize variables */
    int i, j, iter = 0, changed = 1, obsIndex;

    /* observations matrix */
    input_mat = create_data_mat(d,n,p);
    if(input_mat == NULL){
        printf("Couldn't allocate memory for input_mat");
        return NULL;
    }

    /*create centroids matrix.*/
    centroid_mat = create_mat(d, k);
    if(centroid_mat == NULL){
        printf("Couldn't allocate memory for centroid_mat");
        free_mat_object(input_mat);
        return NULL;
    }

    /*initialize the centroids for the first k observations using the info in kClusters.*/
    for(i=0; i<k; i++){
        obsIndex = kClusters[i];
        for (j=0; j<d; j++){
            set_item(centroid_mat,i,j,get_item(input_mat, obsIndex, j));
        }
    }

    /* STEP 2 OF THE KMEANS ALGORITHM */

    /*Allocate memory clusters array to assign each observation xi to Sj.*/
    a_clusters = calloc(n,sizeof(int));
    if(a_clusters == NULL) {
        printf("Failed to allocate memory for c array a_clusters\n");
        free_mat_object(input_mat);
        free_mat(centroid_mat);
        return NULL;
    }

    /*create temp centroid matrix for stage 2.b.*/
    tempcentroid_mat = create_mat(d, k);
    if(tempcentroid_mat == NULL){
        printf("Couldn't allocate memory for tempcentroid_mat");
        free_mat_object(input_mat);
        free_mat(centroid_mat);
        free(a_clusters);
        return NULL;
    }

    /*Allocate memory cluster sum of observations vector. */
    a_clustersum= calloc(k,sizeof(double));
    if(a_clustersum == NULL) {
        printf("Failed to allocate memory for c array a_clustersum\n");
        free_mat_object(input_mat);
        free_mat(centroid_mat);
        free(a_clusters);
        free_mat(tempcentroid_mat);
        return NULL;
    }

    /* THE STEP 2 ITERATIONS */
    while(iter<MAX_ITER && changed ==1){

        /*2.a - Assign each observation xi to cluster Sj, that minimizes the distance between xi to the the centroid. */
        assign_clusters(input_mat, centroid_mat, a_clusters, k, n, d);

        /*2.b */
        changed = 0;

        /*initialize the centroid mat and the cluster sum array to 0; */
        zeroVector(tempcentroid_mat->data, k*d);
        zeroVector(a_clustersum, k);

        /*initialize the tamp centroid mat and the cluster sum array with the new values of 2.a. */
        for(i=0; i<n; i++){
            vectorSum(get_row(tempcentroid_mat, a_clusters[i]),get_row(input_mat, i),d);
            a_clustersum[a_clusters[i]]++;
        }

        /*use scalar func to update the centorids (a_tempcentroid with a_clustersum) */
        for(i=0; i<k ;i++){
            scalarDiv(get_row(tempcentroid_mat, i), d, a_clustersum[i]);
        }

        /*compare the temp centroid mat with the curr centroid mat --> change the value of curr centroid mat. */
        if (vectorCompare(tempcentroid_mat->data, centroid_mat->data, k*d)==0){
            copyValues(centroid_mat->data, tempcentroid_mat->data,k*d);
            changed=1;
        }

        /* advance the while loop in MAX_ITER */
        iter++;
    }

    /* free allocated arrays */
    free_mat_object(input_mat);
    free_mat(centroid_mat);
    free_mat(tempcentroid_mat);
    free(a_clustersum);

    /* return the cluster array */
    return a_clusters;
}

/* a function that receives Pyobjects and prepares them for c_kmeans() */
static PyObject* kmeans(PyObject *self, PyObject *args)
{
    /* assign args to variables */
    PyObject *obsMat, *item, *list, *initializedClusters, *result;
    int K, N, d, i, MAX_ITER;
    double *p;
    int kCounter = 0;
    int pCounter = 0;
    int obsMatj, obsMatk, kitem;
    double ditem;
    int *kClusters;
    int *finalClusters;

    /* error handling */
    if(!PyArg_ParseTuple(args, "OOiiii", &obsMat, &initializedClusters, &K, &N, &d, &MAX_ITER)) {
        return NULL;
    }

    if (!PyList_Check(obsMat))
        return NULL;

    if (!PyList_Check(initializedClusters))
        return NULL;

    /* transform the Pyobject obsMat into a c array named p */
    p = calloc(N*d,sizeof(double));
    if(p == NULL) {
        printf("Failed to allocate memory for c array p\n");
        return NULL;
    }
    for (obsMatk = 0; obsMatk<N; obsMatk++ ){
        list = PyList_GetItem(obsMat, obsMatk);
        for ( obsMatj = 0; obsMatj < d; obsMatj ++) {
            item = PyList_GetItem(list, obsMatj);
            ditem = PyFloat_AsDouble(item);
            *(p+pCounter) = ditem;
            pCounter++;
        }
    }

    /* transform the Pyobject initializedClusters into a c array named kClusters */
    kClusters = calloc(K,sizeof(int));
    if(kClusters == NULL) {
        free(p);
        printf("Failed to allocate memory for c array kClusters\n");
        return NULL;
    }

    for ( kCounter = 0; kCounter < K; kCounter ++) {
            item = PyList_GetItem(initializedClusters, kCounter);
            kitem = PyLong_AsLong(item);
            *(kClusters + kCounter) = kitem;
        }

    /* call the c function that performs kmeans step 2 */
    finalClusters = c_kmeans(p, kClusters, K, N, d, MAX_ITER);

    if(finalClusters == NULL) {
        return NULL;
    }

    /* convert the result into a returnable object */
    result = PyList_New(N);
    for (i = 0; i<N; i++){
        item = Py_BuildValue("i",finalClusters[i]);
        PyList_SetItem(result, i, item);
    }

    /* free allocated arrays */
    free(p);
    free(kClusters);

    /* return the final clusters */
    return result;
}


/* C_PYTHON API */

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }

static PyMethodDef _methods[] = {
    FUNC(METH_VARARGS, kmeans, "calculate the kmeans and return the result"),
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}



