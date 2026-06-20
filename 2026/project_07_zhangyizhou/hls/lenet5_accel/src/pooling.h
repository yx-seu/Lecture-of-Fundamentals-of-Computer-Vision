/*
 * pooling.h — 2x2 Max Pooling (header-only for Vitis HLS)
 */
#pragma once
#include "types.h"

template<int CH, int IN_H, int IN_W, int OUT_H, int OUT_W>
void maxpool_layer(
    data_t in_fmap[CH * IN_H * IN_W],
    data_t out_fmap[CH * OUT_H * OUT_W]
) {
#pragma HLS INLINE off

    for (int c = 0; c < CH; c++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=16
        for (int oy = 0; oy < OUT_H; oy++) {
            for (int ox = 0; ox < OUT_W; ox++) {
#pragma HLS PIPELINE II=1

                int iy = oy * 2;
                int ix = ox * 2;

                data_t v00 = in_fmap[c * IN_H * IN_W + (iy+0) * IN_W + (ix+0)];
                data_t v01 = in_fmap[c * IN_H * IN_W + (iy+0) * IN_W + (ix+1)];
                data_t v10 = in_fmap[c * IN_H * IN_W + (iy+1) * IN_W + (ix+0)];
                data_t v11 = in_fmap[c * IN_H * IN_W + (iy+1) * IN_W + (ix+1)];

                data_t max_h0 = (v00 > v01) ? v00 : v01;
                data_t max_h1 = (v10 > v11) ? v10 : v11;
                data_t max_val = (max_h0 > max_h1) ? max_h0 : max_h1;

                out_fmap[c * OUT_H * OUT_W + oy * OUT_W + ox] = max_val;
            }
        }
    }
}
