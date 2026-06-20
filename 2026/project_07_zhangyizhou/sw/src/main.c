/*
 * main.c — Batch test 100 MNIST images
 */
#include "xparameters.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_cache.h"

#define BASE   XPAR_LENET5_0_BASEADDR
#define IMG    0x400
#define RES    0x10

#include "test_batch.h"

int main(void) {
    Xil_ICacheEnable();
    Xil_DCacheDisable();
    Xil_Out32(0xF8000008, 0xDF0D);
    uint32_t c = Xil_In32(0xF8000170);
    if (!(c & (1 << 8))) Xil_Out32(0xF8000170, c | (1 << 8));
    Xil_Out32(0xF8000004, 0x767B);

    xil_printf("\r\n===== Batch Test (%d images) =====\r\n", TEST_COUNT);

    int correct = 0, per_digit[10] = {0}, total[10] = {0};

    for (int t = 0; t < TEST_COUNT; t++) {
        while (!(Xil_In32(BASE + 0x00) & 4));

        const signed char *img = test_imgs[t];
        for (int i = 0; i < 1024; i += 4) {
            uint32_t w = ((uint32_t)(uint8_t)img[i+0])       |
                         ((uint32_t)(uint8_t)img[i+1]) << 8  |
                         ((uint32_t)(uint8_t)img[i+2]) << 16 |
                         ((uint32_t)(uint8_t)img[i+3]) << 24;
            Xil_Out32(BASE + IMG + i, w);
        }
        __asm__ volatile("dsb" ::: "memory");

        Xil_Out32(BASE + 0x00, 1);
        int to = 0;
        while (!(Xil_In32(BASE + 0x00) & 2))
            if (++to > 20000000) { xil_printf(" T/O@%d", t); while(1); }

        uint32_t r0=Xil_In32(BASE+RES), r1=Xil_In32(BASE+RES+4), r2=Xil_In32(BASE+RES+8);
        int s[10], d=0;
        s[0]=(int8_t)(r0&0xFF); s[1]=(int8_t)((r0>>8)&0xFF);
        s[2]=(int8_t)((r0>>16)&0xFF); s[3]=(int8_t)((r0>>24)&0xFF);
        s[4]=(int8_t)(r1&0xFF); s[5]=(int8_t)((r1>>8)&0xFF);
        s[6]=(int8_t)((r1>>16)&0xFF); s[7]=(int8_t)((r1>>24)&0xFF);
        s[8]=(int8_t)(r2&0xFF); s[9]=(int8_t)((r2>>8)&0xFF);
        for(int i=1;i<10;i++) if(s[i]>s[d]) d=i;

        int lbl = test_labels[t];
        total[lbl]++;
        if (d == lbl) { correct++; per_digit[lbl]++; }
    }

    xil_printf("\r\nAccuracy: %d/%d = %d%%\r\n", correct, TEST_COUNT, correct*100/TEST_COUNT);
    for (int i = 0; i < 10; i++)
        if (total[i] > 0)
            xil_printf("  %d: %d/%d = %d%%\r\n", i, per_digit[i], total[i], per_digit[i]*100/total[i]);
    xil_printf("======================\r\n");
    while(1);
    return 0;
}
