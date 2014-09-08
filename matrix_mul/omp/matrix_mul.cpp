/*
 
 Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**** Modified by Cheng Zhang, Zhe Qian
 **** Last Modification date: Feb 5th
 **** Modified in: line 36, line 43 to 58
 ****/
#include <omp.h>
#include "matrix_mul.h"

namespace omp
{
    void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension ){
        unsigned int row0, row1, rowk;
        float acc00, acc01, acc10, acc11;
        unsigned int i = 0, j, k;
        
        /**** Modification: add pragma omp parallel for statement
         **** to enable parallet computing
         ****/
        #pragma omp parallel for private(i,j,k,row0,row1,rowk,acc00,acc01,acc10,acc11) firstprivate(sq_dimension) shared(sq_matrix_result,sq_matrix_1,sq_matrix_2) schedule(static)
      
        for (i = 0; i < sq_dimension - 1; i+= 2) {
                row0 = i * sq_dimension;
                row1 = (i + 1) * sq_dimension;
                /****  Modification: Using 2 * 2 cache block instead 1 block
                 ****/
                for(j = 0; j < sq_dimension - 1; j+= 2){
                    acc00 = acc01 = acc10 = acc11 = 0;
         
                    for (k = 0; k < sq_dimension; k++){
                        rowk = k * sq_dimension;
                        acc00 += sq_matrix_1[row0 + k] * sq_matrix_2[rowk + j];
                        acc01 += sq_matrix_1[row0 + k] * sq_matrix_2[rowk + j + 1];
                        acc10 += sq_matrix_1[row1 + k] * sq_matrix_2[rowk + j];
                        acc11 += sq_matrix_1[row1 + k] * sq_matrix_2[rowk + j + 1];
                    }
                    sq_matrix_result[row0 + j] = acc00;
                    sq_matrix_result[row0 + j + 1] = acc01;
                    sq_matrix_result[row1 + j] = acc10;
                    sq_matrix_result[row1 + j + 1] = acc11;
                    
                }
                /****Last column may not be able to use 2 * 2 cache block
                 **** Instead, we use 1 * 2 cache block
                 ****/
                for (; j < sq_dimension; j ++){
                    acc00 = 0;
                    acc01 = 0;
                    for (k = 0; k < sq_dimension; k++){
                        acc00 += sq_matrix_1[row0 + k] * sq_matrix_2[k*sq_dimension + j];
                        acc01 += sq_matrix_1[row1 + k] * sq_matrix_2[k*sq_dimension + j];
                    }
                    sq_matrix_result[row0 + j] = acc00;
                    sq_matrix_result[row1 + j] = acc01;
                }
                
        }// End of parallel region
        /*** In case sq_dimension is odd number, we caclulate
         *** the result for the last row again.
         ***/
        for (i = sq_dimension - 1; i < sq_dimension; i++) {
            for (j = 0; j < sq_dimension; j ++){
                sq_matrix_result[i*sq_dimension + j] = 0;
                for (k = 0; k < sq_dimension; k++){
                    sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
                }
            }
        }
    }
    
}
