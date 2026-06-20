/*
 * conv2d.h — Parameterized 2D Convolution (resource-constrained version)
 *
 * Optimized for xc7z010 (80 DSP, 60 BRAM, 17.6K LUTs).
 * Uses sequential MAC processing with limited parallelism.
 */
#pragma once
#include "types.h"

template<int IN_CH, int IN_H, int IN_W, int OUT_CH, int OUT_H, int OUT_W, int K, int REQ_M>
void conv2d_layer(
    data_t in_fmap[IN_CH * IN_H * IN_W],
    const data_t weights[OUT_CH * IN_CH * K * K],
    const bias_t biases[OUT_CH],
    data_t out_fmap[OUT_CH * OUT_H * OUT_W]
) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=mul limit=8 type=core

    conv2d_outer:
    for (int co = 0; co < OUT_CH; co++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=120

    conv2d_row:
        for (int oy = 0; oy < OUT_H; oy++) {
        conv2d_col:
            for (int ox = 0; ox < OUT_W; ox++) {
                accum_t acc = 0;

            conv2d_ci:
                for (int ci = 0; ci < IN_CH; ci++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                conv2d_ky:
                    for (int ky = 0; ky < K; ky++) {
                    conv2d_kx:
                        for (int kx = 0; kx < K; kx++) {
                            int iy = oy + ky;
                            int ix = ox + kx;

                            if (iy < IN_H && ix < IN_W) {
                                data_t pixel = in_fmap[ci * IN_H * IN_W + iy * IN_W + ix];
                                data_t w = weights[co * IN_CH * K * K + ci * K * K + ky * K + kx];
                                acc += (accum_t)pixel * (accum_t)w;
                            }
                        }
                    }
                }

                acc += biases[co];
                out_fmap[co * OUT_H * OUT_W + oy * OUT_W + ox] = requantize<REQ_M>(acc);
            }
        }
    }
}
