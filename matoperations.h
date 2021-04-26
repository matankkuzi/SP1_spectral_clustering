/** @file matoperations.h
 *
 * @brief This module contains the mat struct and the
            matrix operations that are the building blocks
            of the K-means algorithm.
 */

/* the matrix struct */
typedef struct mat
{
    int rows;
    int cols;
    double *data;
} mat;

/* create a matrix object */
mat *create_mat(const int rows, const int cols);

/* create a matrix object and set its data */
mat *create_data_mat(const int rows, const int cols, double *data);

/* free the matrix object and its data array */
void free_mat(mat *matrix);

/* free the matrix object - do not free its data array */
void free_mat_object(mat *matrix);

/* point to a specific row of the matrix */
double *get_row(const mat *matrix, const int i);

/* get item from a specific cell in the matrix */
double get_item(const mat *matrix, const int i, const int j);

/* set value for a specific cell in the matrix */
void set_item(const mat *matrix, const int i, const int j, const double x);

/* initialize a vector with zeros*/
void zeroVector(double *vec, int len);

/*copy values of matrix to another*/
void copyValues(double *mat1,double *mat2,int len);

/*distance between two vectors .*/
double distance(double *vec1, double *vec2, int len);

/*Sum of two vectors*/
void vectorSum(double *vec1, double *vec2, int len);

/*Divide a vector by a scalar*/
void scalarDiv(double *vec, int len, double num);

/*Compare two vectors. return 1 if they are the same. else return 0.*/
int vectorCompare(double *vec1, double *vec2, int len);


/*** end of file ***/