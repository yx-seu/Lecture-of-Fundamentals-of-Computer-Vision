/*
 * fc.h — Fully Connected Layer (resource-constrained)
 */
#pragma once
#include "types.h"

template<int IN_DIM, int OUT_DIM, bool USE_RELU, int REQ_M>
void fc_layer(
    data_t in_fmap[IN_DIM],
    const data_t weights[OUT_DIM * IN_DIM],
    const bias_t biases[OUT_DIM],
    data_t out_fmap[OUT_DIM]
) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=mul limit=8 type=core

    for (int j = 0; j < OUT_DIM; j++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=120

        accum_t acc = 0;

        for (int i = 0; i < IN_DIM; i++) {
            acc += (accum_t)in_fmap[i] * (accum_t)weights[j * IN_DIM + i];
        }

        acc += biases[j];

        if (USE_RELU) {
            out_fmap[j] = requantize<REQ_M>(acc);
        } else {
            out_fmap[j] = requantize_no_relu<REQ_M>(acc);
        }
    }
}
