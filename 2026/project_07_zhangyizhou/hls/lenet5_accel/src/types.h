/*
 * types.h — Fixed-point type definitions for LeNet-5 HLS accelerator
 *
 * Data format:
 *   - Weights:  int8 (Q0.7), symmetric, range [-1.0, +1.0)
 *   - Features: int8 (Q5.3), unsigned for ReLU output, range [0, 15.875]
 *   - Accum:    int16, accumulation of weight * feature products
 *
 * Scaling convention:
 *   float_val = int_val * scale
 *   After MAC accumulation: acc >> SHIFT to convert back to Q5.3
 */

#pragma once

#include <ap_int.h>
#include <stdint.h>

// === Fixed-point types ===
typedef ap_int<8>   data_t;    // Feature maps & weights (int8)
typedef ap_uint<8>  udata_t;   // Unsigned pixel input
typedef ap_int<32>  accum_t;   // Accumulator (int32, wide enough for full sum)
typedef ap_int<32>  bias_t;    // Bias (int32)

// === Quantization parameters ===
#define WEIGHT_FRAC_BITS  7     // Q0.7 weight format
#define ACT_FRAC_BITS     3     // Q5.3 activation format
#define REQ_SHIFT         20    // Requantization shift (M * 2^-N) per layer
#define C1_REQ_M          5600  // C1: w_scale*a_in/a_out * 2^20
#define C3_REQ_M          1850  // C3
#define C5_REQ_M          1700  // C5
#define F6_REQ_M          2558  // F6
#define OUT_REQ_M         1941  // OUT

// Scale factors (from Python quantization)
// These are defined per-layer to handle different weight/activation scales
// C1: w_scale=0.005340, a_scale=0.585189
// C3: w_scale=0.002666, a_scale=0.884023
// C5: w_scale=0.001740, a_scale=0.948717
// F6: w_scale=0.002337, a_scale=0.908750
// OUT: w_scale=0.003407, a_scale=1.672747

// === LeNet-5 layer dimensions ===
#define INPUT_H     32
#define INPUT_W     32
#define INPUT_CH    1

// C1: Conv2d(1,6,5) -> 6x28x28
#define C1_OUT_CH   6
#define C1_KERNEL   5
#define C1_OUT_H    28
#define C1_OUT_W    28

// S2: MaxPool(2,2) -> 6x14x14
#define S2_POOL     2
#define S2_OUT_H    14
#define S2_OUT_W    14

// C3: Conv2d(6,16,5) -> 16x10x10
#define C3_IN_CH    6
#define C3_OUT_CH   16
#define C3_KERNEL   5
#define C3_OUT_H    10
#define C3_OUT_W    10

// S4: MaxPool(2,2) -> 16x5x5
#define S4_POOL     2
#define S4_OUT_H    5
#define S4_OUT_W    5

// C5: Conv2d(16,120,5) -> 120x1x1
#define C5_IN_CH    16
#define C5_OUT_CH   120
#define C5_KERNEL   5
#define C5_OUT_H    1
#define C5_OUT_W    1

// F6: Linear(120,84)
#define F6_IN_DIM   120
#define F6_OUT_DIM  84

// OUT: Linear(84,10)
#define OUT_IN_DIM  84
#define OUT_OUT_DIM 10

// === Hardware parameters ===
#define MAC_PARALLEL    8      // Number of parallel MAC units
#define MAX_LINE_WIDTH  32     // Max line buffer width (C1: 32)

// === Helper: requantize with per-layer multiplier ===
// result = (acc * M) >> N, then clip + ReLU
template<int M>
static inline data_t requantize(accum_t acc) {
#pragma HLS INLINE
    accum_t scaled = (acc * (accum_t)M) >> REQ_SHIFT;
    if (scaled > 127) return 127;
    if (scaled < 0)   return 0;    // ReLU
    return (data_t)scaled;
}

// No-ReLU version for output layer
template<int M>
static inline data_t requantize_no_relu(accum_t acc) {
#pragma HLS INLINE
    accum_t scaled = (acc * (accum_t)M) >> REQ_SHIFT;
    if (scaled > 127)  return 127;
    if (scaled < -128) return -128;
    return (data_t)scaled;
}
