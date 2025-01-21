/**
 * This kernel function sums two arrays of integers and returns its result
 * through a third array.
 **/

 __kernel void sumArrays(__global int const * restrict a, __global int const* restrict b, __global int* c){
     int index = get_global_id(0);
     c[index] = a[index] + b[index];
 }