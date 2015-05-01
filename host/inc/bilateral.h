#ifndef BILATERAL_H
#define BILATERAL_H

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



//! @param imgname The string literal of the image
//! @param radius The radius of the radius
//! @param sigma The sigma parameter for the bilateral kernel
//! @return Returns true for success or false for failure
char b_filter_FPGA(char* ,uint32_t ,float );
char b_filter_ARM(char*, uint32_t, float);

#endif
