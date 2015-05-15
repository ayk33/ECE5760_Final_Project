#ifndef GAUSSIAN_H
#define GAUSSIAN_H

//uncomment if you want to measure the communication overhead instead of normally running the program
//#define MEASURE_COMM_OVERHEAD

#include "bitmap.h"
#ifndef _MSC_VER
    #define __STDC_FORMAT_MACROS //also request the printf format macros
    #include <inttypes.h>
#else//msvc does not have the C99 standard header so we gotta define them explicitly here, since they do have some similar types
    typedef unsigned __int8 uint8_t;
    typedef __int8  int8_t;
    typedef unsigned __int16 uint16_t;
    typedef __int16 int16_t;
    typedef unsigned __int32 uint32_t;
    typedef __int32 int32_t;
    typedef unsigned __int64 uint64_t;
    typedef __int64 int64_t;
#endif


char gaussian_blur_ARM(char* imgname,uint32_t size,float sigma);
char gaussian_blur_FPGA(char* imgname,uint32_t size,float sigma);

#endif
