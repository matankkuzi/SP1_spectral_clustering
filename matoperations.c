/*
 ============================================================================
 Name        : matoperations.c
 Description : This file contains the mat struct and the
                matrix operations that are the building blocks
                of the K-means algorithm.
 ============================================================================
 */

#include "matoperations.h"
#include <stdlib.h>
#include <stdio.h>


/* create a matrix object */
mat *create_mat(const int rows, const int cols)
{
    mat *matrix = calloc(1, sizeof(mat));
    if (!matrix) {
        printf("Failed to allocate memory for matrix");
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->data = calloc(rows * cols, sizeof(double));
    if (!matrix->data) {
        free_mat_object(matrix);
        printf("Data didn't transfer porperly");
        return NULL;
    }
    return matrix;
}


/* create a matrix object and set its data */
mat *create_data_mat(const int rows, const int cols, double *data)
{
    mat *matrix = calloc(1, sizeof(mat));
    if (!matrix) {
        printf("Failed to allocate memory for matrix");
        return NULL;
    }

    matrix->rows = rows;
    matrix->cols = cols;

    matrix->data = data;
    if (!matrix->data) {
        printf("Data didn't transfer porperly");
    }
    return matrix;
}


/* free the matrix object and its data array */
void free_mat(mat *matrix)
{
    if(matrix->data != NULL) free(matrix->data);
    if(matrix != NULL) free(matrix);
}


/* free the matrix object - do not free its data array */
void free_mat_object(mat *matrix)
{
    if(matrix != NULL) free(matrix);
}


/* point to a specific row of the matrix */
double *get_row(const mat *matrix, const int i)
{
    return matrix->data + i * matrix->rows;
}


/* get item from a specific cell in the matrix */
double get_item(const mat *matrix, const int i, const int j)
{
    return *(matrix->data + i * matrix->rows + j);
}


/* set value for a specific cell in the matrix */
void set_item(const mat *matrix, const int i, const int j, const double x)
{
    *(matrix->data + i * matrix->rows + j) = x;
}


/* initialize a vector with zeros*/
void zeroVector(double *vec, int len){
    int i;
    for (i = 0; i< len; i++)
        *(vec+i) = 0;
}

/*copy values of matrix to another*/
void copyValues(double *mat1,double *mat2,int len){
    int i;
    for(i=0; i<len; i++)
        mat1[i] = mat2[i];
}

/*distance between two vectors .*/
double distance(double *vec1, double *vec2, int len)
{
    double distance1 = 0;
    int i;
    for(i=0; i<len; i++){
        distance1 += (*(vec1+i) - *(vec2+i))*(*(vec1+i) - *(vec2+i));
    }
    return distance1;
}

/*Sum of two vectors*/
void vectorSum(double *vec1, double *vec2, int len){
    int i;
    for (i=0; i<len; i++)
        *(vec1+i) +=  *(vec2 + i);
}

/*Divide a vector by a scalar*/
void scalarDiv(double *vec, int len, double num){
    int i;
    for (i=0; i< len; i++)
        *(vec+i) = *(vec+i)/num;
}

/*Compare two vectors. return 1 if they are the same. else return 0.*/
int vectorCompare(double *vec1, double *vec2, int len){
    int res = 1, i;
    for (i=0; i<len; i++)
        if (*(vec1+i)!=*(vec2+i))
            res = 0;
    return res;
}