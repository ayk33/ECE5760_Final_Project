//The Gaussian blur function that runs on the gpu
#define POW2(a) ((a) * (a))
#define PI_ 3.14159265359f


__kernel void gaussian_blur(__global const unsigned char *image, const int W,const int H,const int size, __global unsigned char* newImg, const float sigma) 
{ 
	unsigned int x,y,imgLineSize;
	float value, sum, gaussian_weight;
	int i,xOff,yOff,center;
    
  //Get the index of the current element being processed
  i = get_global_id(0);
	
  //Calculate some needed variables
	imgLineSize = W*3;
	center = size/2;
	//pass the pixel through the kernel if it can be centered inside it
  
  
  //create the Gaussian kernel
  
	if(i >= imgLineSize*(size-center)+center*3 &&
	   i < W*H*3-imgLineSize*(size-center)-center*3)
	{
		value=0;
    sum = 0; 
		for(y=0;y<size;y++)
        {
          yOff = imgLineSize*(y-center);
          for(x=0;x<size;x++)
          {
            xOff = 3*(x-center);
            gaussian_weight = exp( (((x-center)*(x-center)+(y-center)*(y-center))/(2.0f*sigma*sigma))*-1.0f ) / (2.0f*PI_*sigma*sigma);
            value += gaussian_weight*image[i+xOff+yOff];
            sum += gaussian_weight;
          }
        }
        //normalize
        newImg[i] = value/sum;
	}
	else//if it's in the edge keep the same value
	{
		newImg[i] = image[i];
	}
}
 
__kernel void bilateral_filter(__global const unsigned char *image, const int W, const int H, const int size, const float sigma, __global unsigned char* newImg)
{
    //local variables for iterating through the image
    //and for holding temporary values
    unsigned int x,y,imgLineSize;
    int i, center,yOff,xOff;
    
    float diff_map, gaussian_weight,value, weight, count, center_pix;

    //find the size of one line of the image in bytes and the center of the bilateral filter
    imgLineSize = W*3;
    center = size/2;
    
    //Get local index
    i = get_global_id(0);
    
    //Run the window through all of the image
    if(i >= imgLineSize*(size-center)+center*3 && i < W*H*3-imgLineSize*(size-center)-center*3)
    {
        //Reseting count and value
        count       = 0;
        value       = 0;
        
        //Obtaining windows center pixel 
        center_pix = (float)image[i+imgLineSize*center + center*3];
        
        //Vertical window loop
        for(y=0;y<size;y++)
        {
            //Horizontal window loop
            yOff = imgLineSize*(y-center);
            for(x=0;x<size;x++)
            {
                xOff = 3*(x - center);

                diff_map = exp (-0.5f *(POW2(center_pix - (float)image[i+xOff+yOff])) * sigma); 
                gaussian_weight = exp( - 0.5f * (POW2(x) + POW2(y)) / (size*size));
                
                //printf("diff_map value %f\n", diff_map); 
                weight = gaussian_weight * diff_map;
                value += weight * image[i+xOff+yOff];
                count += weight; 
            }
        }
        newImg[i] = (unsigned char)(value / count);
    }
    else//if it's in the edge keep the same value
    {
      newImg[i] = image[i];
    }
}
