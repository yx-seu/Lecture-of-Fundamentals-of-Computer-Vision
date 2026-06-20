/*
 * lenet5_accel.cpp — s_axilite in_image + separate out_scores
 * in_image at 0x400 (verified working), out_scores as s_axilite array
 */
#include "types.h"
#include "weights.h"
#include "conv2d.h"
#include "pooling.h"
#include "fc.h"

void lenet5_accel(
    data_t  in_image[INPUT_CH * INPUT_H * INPUT_W],
    data_t  out_scores[OUT_OUT_DIM]
) {
#pragma HLS INTERFACE s_axilite port=return     bundle=control
#pragma HLS INTERFACE s_axilite port=in_image   bundle=control
#pragma HLS INTERFACE s_axilite port=out_scores bundle=control

    static data_t buf[INPUT_CH*INPUT_H*INPUT_W];
#pragma HLS BIND_STORAGE variable=buf type=RAM_T2P impl=BRAM
    static data_t buf_c1[C1_OUT_CH*C1_OUT_H*C1_OUT_W];
#pragma HLS BIND_STORAGE variable=buf_c1 type=RAM_T2P impl=BRAM
    static data_t buf_s2[C1_OUT_CH*S2_OUT_H*S2_OUT_W];
#pragma HLS BIND_STORAGE variable=buf_s2 type=RAM_T2P impl=BRAM
    static data_t buf_c3[C3_OUT_CH*C3_OUT_H*C3_OUT_W];
#pragma HLS BIND_STORAGE variable=buf_c3 type=RAM_T2P impl=BRAM
    static data_t buf_s4[C3_OUT_CH*S4_OUT_H*S4_OUT_W];
#pragma HLS BIND_STORAGE variable=buf_s4 type=RAM_T2P impl=BRAM
    static data_t buf_c5[C5_OUT_CH*C5_OUT_H*C5_OUT_W];
#pragma HLS BIND_STORAGE variable=buf_c5 type=RAM_T2P impl=BRAM
    static data_t buf_f6[F6_OUT_DIM];
#pragma HLS BIND_STORAGE variable=buf_f6 type=RAM_T2P impl=BRAM

    // Copy from AXI registers to local BRAM
    copy_in: for (int i = 0; i < INPUT_CH*INPUT_H*INPUT_W; i++) {
#pragma HLS PIPELINE
        buf[i] = in_image[i];
    }

    conv2d_layer<INPUT_CH,INPUT_H,INPUT_W,C1_OUT_CH,C1_OUT_H,C1_OUT_W,C1_KERNEL,C1_REQ_M>(
        buf, c1_w, c1_b, buf_c1);
    maxpool_layer<C1_OUT_CH,C1_OUT_H,C1_OUT_W,S2_OUT_H,S2_OUT_W>(buf_c1, buf_s2);
    conv2d_layer<C3_IN_CH,S2_OUT_H,S2_OUT_W,C3_OUT_CH,C3_OUT_H,C3_OUT_W,C3_KERNEL,C3_REQ_M>(
        buf_s2, c3_w, c3_b, buf_c3);
    maxpool_layer<C3_OUT_CH,C3_OUT_H,C3_OUT_W,S4_OUT_H,S4_OUT_W>(buf_c3, buf_s4);
    conv2d_layer<C5_IN_CH,S4_OUT_H,S4_OUT_W,C5_OUT_CH,C5_OUT_H,C5_OUT_W,C5_KERNEL,C5_REQ_M>(
        buf_s4, c5_w, c5_b, buf_c5);
    fc_layer<F6_IN_DIM,F6_OUT_DIM,true,F6_REQ_M>(buf_c5, f6_w, f6_b, buf_f6);

    // OUT layer inline — write directly to out_scores
    for (int j = 0; j < OUT_OUT_DIM; j++) {
#pragma HLS UNROLL
        accum_t acc = 0;
        for (int i = 0; i < OUT_IN_DIM; i++) {
#pragma HLS PIPELINE
            acc += (accum_t)buf_f6[i] * (accum_t)out_w[j * OUT_IN_DIM + i];
        }
        acc += out_b[j];
        out_scores[j] = requantize_no_relu<OUT_REQ_M>(acc);
    }
}
