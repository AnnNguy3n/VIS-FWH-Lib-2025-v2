#pragma once
#include "../DoubleYearThreshold/DoubleYearThresholdKernel.cu"
#include "../../Generator/HomoPoly/CUDA/HomoPolyMethod.cu"


__device__ __forceinline__ void _triple_year_threshold_investing(double *weight, double threshold, int t_idx, double *result,
    double INTEREST, int *INDEX, double *PROFIT, int *SYMBOL, int *BOOL_ARG, int index_size, int num_cycle){
    int reason;
    double Geo2 = 0, Har2 = 0;
    int start, end, end2, count, k, sym, s, rs_idx, s3, end3;
    double temp, n;
    bool check;

    check = false;
    start = INDEX[index_size - 3];
    end = INDEX[index_size - 2];
    end2 = INDEX[index_size - 1];
    for (k=start; k<end; k++){
        if (weight[k] > threshold){
            check = true;
            break;
        }
    }
    if (!check) reason = 2;
    else {
        for (k=end; k<end2; k++){
            if (weight[k] > threshold){
                check = false;
                break;
            }
        }
        if (!check) reason = 0;
        else reason = 1;
    }

    for (int i=index_size-4; i>0; i--){
        start = INDEX[i];
        end = INDEX[i+1];
        end2 = INDEX[i+2];
        end3 = INDEX[i+3];
        temp = 0;
        count = 0;
        check = false;
        for (k=start; k<end; k++){
            if (weight[k] > threshold){
                check = true;
                if (!BOOL_ARG[k]) continue;

                if (reason == 2){
                    count ++;
                    temp += PROFIT[k];
                }
                else {
                    sym = SYMBOL[k];
                    s = binary_symbol_search(SYMBOL, end, end2, sym);
                    if (s != -1 && weight[s] > threshold){
                        if (reason == 1){
                            count++;
                            temp += PROFIT[k];
                        }
                        else {
                            s3 = binary_symbol_search(SYMBOL, end2, end3, sym);
                            if (s3 != -1 && weight[s3] > threshold){
                                count++;
                                temp += PROFIT[k];
                            }
                        }
                    }
                }
            }
        }

        if (!count){
            Geo2 += log(INTEREST);
            Har2 += 1.0 / INTEREST;
        } else {
            temp /= count;
            Geo2 += log(temp);
            Har2 += 1.0 / temp;
        }

        if (!check) reason = 2;
        else {
            if (reason == 2) reason = 1;
            else reason = 0;
        }

        if (i <= num_cycle && t_idx+1 >= i){
            rs_idx = num_cycle - i;
            n = index_size - 3 - i;
            result[2*rs_idx] = exp(Geo2/n);
            result[2*rs_idx+1] = n / Har2;
        }
    }
}


__global__ void triple_year_threshold_investing(double *weights, double *thresholds, double *results, int num_array, int num_threshold,
    int length, int num_cycle, double INTEREST, int *INDEX, double *PROFIT, int *SYMBOL, int *BOOL_ARG, int index_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_array*num_threshold){
        int ix = index % num_threshold;
        int iy = index / num_threshold;
        _triple_year_threshold_investing(
            weights + iy*length,
            thresholds[iy*num_threshold + ix],
            ix / __NUM_THRESHOLD_PER_CYCLE__,
            results + iy*num_threshold*num_cycle*2 + ix*num_cycle*2,
            INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG, index_size, num_cycle
        );
    }
}
