/*
 * lenet5_accel.h — Top-level LeNet-5 accelerator
 */
#pragma once
#include "types.h"

void lenet5_accel(
    data_t  in_image[INPUT_CH * INPUT_H * INPUT_W],  // 1x32x32 = 1024
    data_t  out_scores[OUT_OUT_DIM]                   // 10 scores (int8)
);
