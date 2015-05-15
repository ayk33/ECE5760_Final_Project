#include <bitmap.h>
#include <gaussian.h>
#include <bilateral.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>



#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

void cleanup();

using namespace aocl_utils;
int main() 
{
    double units;
    int num_iters = 1;
    //gaussian blur parameters
    uint32_t gaussianSize = 9;
    float gaussianSigma = 4;
    
    //bilateral filter parameters
    uint32_t bilateralRadius = 5;
    float bilateralSigma_squared = .0025;

    char imgname[] = "image1.BMP";//the name of the image file, taken by the arguments
   
    //Running Gaussian Filter on FPGA and timing it.
    double start_time_gaussian_fpga = getCurrentTimestamp();
    printf("\nFilter type: Gaussian - FPGA\n");
    for(int i = 0; i < num_iters; ++i) {
      if(gaussian_blur_FPGA(imgname,gaussianSize,gaussianSigma)==false){
        printf("Failed to call function blur\n");
        return -1;
      }
    }
    double elapsed_time_guassian_fpga = getCurrentTimestamp() - start_time_gaussian_fpga;
    printf("-------------------Guassian Filter FPGA-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_guassian_fpga);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_guassian_fpga / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_guassian_fpga));
    /*
    //Running Gaussian Filter on ARM core and timing it.
    double start_time_gaussian_arm = getCurrentTimestamp();
    printf("\nFilter type: Gaussian - ARM\n");
    for(int i = 0; i < num_iters; ++i) {
      if(gaussian_blur_ARM(imgname,gaussianSize,gaussianSigma)==false){
        printf("Failed to call function blur\n");
        return -1;
      }
    }
    double elapsed_time_guassian_arm = getCurrentTimestamp() - start_time_gaussian_arm;
    printf("-------------------Guassian Filter ARM-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_guassian_arm);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_guassian_arm / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_guassian_arm));
    
    
    //Running Bilateral Filter on FPGA and timing it.
    double start_time_bilateral_fpga = getCurrentTimestamp();
    printf("\nFilter type: Bilateral - FPGA\n");
    for(int i = 0; i < num_iters; ++i) {
      if(b_filter_FPGA(imgname,bilateralRadius,bilateralSigma_squared)==false){
        printf("Failed to call function bilateral_filter\n");
        return -1;
      }
    }
    double elapsed_time_bilateral_fpga = getCurrentTimestamp() - start_time_bilateral_fpga;
    printf("-------------------Bilateral Filter FPGA-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_bilateral_fpga);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_bilateral_fpga / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_bilateral_fpga));
    
    
    //Running Bilateral Filter on ARM core and timing it.
    double start_time_bilateral_arm = getCurrentTimestamp();
    printf("\nFilter type: Bilateral - ARM\n");
    for(int i = 0; i < num_iters; ++i) {
      if(b_filter_ARM(imgname,bilateralRadius,bilateralSigma_squared)==false){
        printf("Failed to call function bilateral_filter\n");
        return -1;
      }
    }
    double elapsed_time_bilateral_arm = getCurrentTimestamp() - start_time_bilateral_arm;
    printf("-------------------Bilateral Filter ARM-------------------\n");
    printf("%d iterations time: %0.3f seconds\n", num_iters, elapsed_time_bilateral_arm);
    printf("Average single iteration time: %0.3f seconds\n", elapsed_time_bilateral_arm / num_iters);
    printf("Throughput = %0.3f FPS\n", (float)(num_iters/elapsed_time_bilateral_arm));
   
    printf("Finished filtering image\n");
    //cleanup(); 
*/
    return 0;
}

void cleanup()
{
  
  //This is here for aocl utils
}
