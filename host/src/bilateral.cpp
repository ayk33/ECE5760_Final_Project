#include <bilateral.h>
#include <gaussian.h>
#include "CL/opencl.h"
#include <stdint.h>
#include <math.h>
#include "AOCLUtils/aocl_utils.h"


#define MAX_SOURCE_SIZE (1048576) //1 MB
#define MAX_LOG_SIZE    (1048576) //1 MB
#define POW2(a) ((a) * (a))

 //Get platform and device information
  static cl_platform_id platform;
  static cl_device_id device;
  static cl_context context;
  static cl_program program;
  static cl_int status;
  static cl_command_queue queue;
  cl_uint num_devices_b;

using namespace aocl_utils;
// Bilateral filter using the arm core
char b_filter_ARM(char* imgname,uint32_t size, float sigma_squared)
{
    uint32_t i,x,y,imgLineSize;
    int32_t center,yOff,xOff;
    float diff_map, gaussian_weight,value, weight, count;
    float center_pix;

    //read the bitmap
    ME_ImageBMP bmp;
    if(meImageBMP_Init(&bmp,imgname)==false)
    {
        printf("Image \"%s\" could not be read as a .BMP file\n",imgname);
        return false;
    }
    
    //find the size of one line of the image in bytes and the center of the bilateral filter
    imgLineSize = bmp.imgWidth*3;
    center = size/2;
    
    //Run the window through all of the image
    for(i = imgLineSize*(size-center)+center*3; i < (bmp.imgHeight*bmp.imgWidth*3)-imgLineSize*(size-center)-center*3;i++)
    {   
        count       = 0.0f;
        value       = 0;
        center_pix = (float)bmp.imgData[i+imgLineSize*center + center*3];
        for(y=0;y<size;y++)
        {
            yOff = imgLineSize*(y-center);
            for(x=0;x<size;x++)
            {
                xOff = 3*(x - center);

                diff_map = exp (-0.5f *(POW2(center_pix - (float)bmp.imgData[i+xOff+yOff])) * sigma_squared); 
                gaussian_weight = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                
                //printf("diff_map value %f\n", diff_map); 
                weight = gaussian_weight * diff_map;
                value += weight * bmp.imgData[i+xOff+yOff];
                count += weight; 
            }
        }
        bmp.imgData[i] = (unsigned char)(value / count);
    }
    //save the image
    char FilteredImage[] = "ARM_Bilateral_Filter.bmp";
    meImageBMP_Save(&bmp,FilteredImage);
    return true;
}

//Bilateral filter the given image using the FPGA
char b_filter_FPGA(char* imgname,uint32_t size,float sigma_squared)
{
    uint32_t imgSize;
    cl_int ret;//the openCL error code/s
    
    //get the image
    ME_ImageBMP bmp;
    meImageBMP_Init(&bmp,imgname);
    imgSize = bmp.imgWidth*bmp.imgHeight*3;
    
    
    //create the pointer that will hold the new (blurred) image data
    unsigned char* newData;
    newData = (unsigned char *)malloc(imgSize);
    
    platform = findPlatform("Altera");
    if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      return false;
    }
  
    // Query the available OpenCL device.
    cl_device_id *devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices_b);
   // printf("Platform: %s\n", getPlatformName(platform).c_str());
   // printf("Found %d device(s)\n", num_devices_b);
    
     // Just use the first device.
    device = devices[0];
    //printf("Using %s\n", getDeviceName(device).c_str());
    delete[] devices;

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Could not create a valid OpenCL context\n");
        return false;
    }
  
    // Create command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    
    // Create memory buffers on the device for the two images
    cl_mem FPGAImg = clCreateBuffer(context,CL_MEM_READ_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
   
    cl_mem FPGANewImg = clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgSize,NULL,&ret);
    if(ret != CL_SUCCESS)
    {
        printf("Unable to create the FPGA image buffer object\n");
        return false;
    }
    
    //Copy the image data kernel to the memory buffer
    if(clEnqueueWriteBuffer(queue, FPGAImg, CL_TRUE, 0,imgSize,bmp.imgData, 0, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error during sending the image data to the OpenCL buffer\n");
        return false;
    }
 
    
    // Create the program using binary already compiled offline using aoc (i.e. the .aocx file)
    std::string binary_file = getBoardBinaryFile("kernel", device);
    //printf("Using AOCX: %s\n", binary_file.c_str());
    
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  
    // build the program
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    cl_kernel kernel = clCreateKernel(program, "bilateral_filter", &ret);
    if(ret != CL_SUCCESS)
    {
        printf("Failed to create the OpenCL Kernel from the built program\n");
        return false;
    }
    // Set the arguments of the kernel
    if(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&FPGAImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGAImg\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 1, sizeof(int), (void *)&bmp.imgWidth) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imageWidth\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 2, sizeof(int), (void *)&bmp.imgHeight) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"imgHeight\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,3,sizeof(int),(void*)&size) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"bilateral size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel,4,sizeof(int),(void*)&sigma_squared) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"bilateral size\" argument\n");
        return false;
    }
    if(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&FPGANewImg) != CL_SUCCESS)
    {
        printf("Could not set the kernel's \"FPGANewImg\" argument\n");
        return false;
    }
    

    ///enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize = imgSize;//The total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize = 64; //The size of one work group
    
    //Enqueue the actual kernel
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize, 0, NULL, NULL);


    ///Read the memory buffer of the new image on the device to the new Data local variable
    ret = clEnqueueReadBuffer(queue, FPGANewImg, CL_TRUE, 0,imgSize, newData, 0, NULL, NULL);

    ///Clean up everything
    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(FPGAImg);
    clReleaseMemObject(FPGANewImg);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    ///save the new image and return success
    bmp.imgData = newData;
    char FilteredImage[] = "FPGA_Bilateral_Filter.bmp";
    meImageBMP_Save(&bmp,FilteredImage);

    free(newData);
    return true;
}
