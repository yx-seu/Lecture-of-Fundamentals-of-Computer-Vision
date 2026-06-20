/* tb for solution15 — s_axilite in_image + out_scores */
#include <stdio.h>
#include <math.h>
#include "../src/types.h"
#include "test_image_0.h"
#include "test_image_3.h"
#include "test_image_7.h"

extern void lenet5_accel(data_t in_image[1024], data_t out_scores[10]);

static void preprocess(const uint8_t img[32][32], data_t out[1024]) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) {
            float v = img[i][j]/255.0f, n = (v-0.1307f)/0.3081f;
            int q = (int)(n/0.585f + 0.5f);
            if(q>127)q=127; if(q<-128)q=-128;
            out[i*32+j]=(data_t)q;
        }
}

int main() {
    data_t buf[1024], res[10]; int p=0;
    auto test=[&](auto img, int exp, const char* name){
        preprocess(img, buf); lenet5_accel(buf, res);
        int d=0; for(int i=1;i<10;i++) if(res[i]>res[d]) d=i;
        printf("Test %s: pred=%d exp=%d -> %s\n",name,d,exp,(d==exp)?"PASS":"FAIL");
        return (d==exp)?1:0;
    };
    p+=test(test_image_0,0,"Digit 0");
    p+=test(test_image_3,3,"Digit 3");
    p+=test(test_image_7,7,"Digit 7");
    printf("\n%d/3\n",p); return (p==3)?0:1;
}
