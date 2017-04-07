#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>
#include "kdtree.h"

static float dist_sq( float *a1, float *a2, int dims  );


int IDWSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output)
{

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int output_height = output->size[1];
  int output_width = output->size[2];
  int inputImages_channels = inputImages->size[3];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[1];
  int output_strideWidth = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];


  float *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;
  float point_list[output_height][output_width][2];
  float grid_list[output_height][output_width][2];
  void *kd;

  for(b=0; b < batchsize; b++)
  {
    kd = kd_create(2);

    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        float yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        float xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];
        point_list[yOut][xOut][0] = yOut;
        point_list[yOut][xOut][1] = xOut;
        grid_list[yOut][xOut][0] = yf;
        grid_list[yOut][xOut][1] = xf;
        //printf("yf,%f xf %f\n", yf, xf);

        const int inAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        //const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        //printf("%.3f, %.3f inserted\n", yf, xf);

        const float pt[] = {yf, xf};
        kd_insertf(kd, pt, inAddress);

        // get the weights for interpolation
        /*int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);

        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
           if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
           if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
           if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

           v = xWeightTopLeft * yWeightTopLeft * inTopLeft
             + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
             + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
             + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

           output_data[outAddress + t] = v;
        }
        */
      }
    }


    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        //real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        //real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        float y = (float)(yOut) / (float)(output_height) * 2 - 1;
        float x = (float)(xOut) / (float)(output_width) * 2 - 1;

        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        void *result_set;
        float pt[] = {y, x};
        result_set = kd_nearest_rangef(kd, pt, 3 * 1/(float)(output_width));
        float pos[2], dist;
        int pch;
        float * dists;
        float sum_dist;
        int npoints = kd_res_size(result_set);
        //printf("%d points found\n",npoints);

        dists = (float *)malloc(npoints * sizeof(float));

        int idx;
        idx = 0;
        sum_dist = 0;
        while( !kd_res_end( result_set  )  ) {
                /* get the data and position of the current result item */
                pch = (int)kd_res_itemf( result_set, pos  );
                    /* compute the distance of the current result from the pt */
                dist = sqrt( dist_sq( pt, pos, 2  )  );
                //printf( "node at (%.3f, %.3f, %.3f) is %.3f away from (%.3f, %.3f, %.3f) and has data=%d\n",
                //                        pos[0], pos[1], pos[2], dist, pt[0], pt[1], pt[2],  pch );
                kd_res_next( result_set );
                dists[idx] = 1/(dist + 1e-6);
                sum_dist += 1/(dist + 1e-6);
                idx+=1;
        }
        //interpolation happens here
        //
        int t;
        for (t=0; t<inputImages_channels;t++)
            output_data[outAddress +t] = 0;
        idx = 0;

        kd_res_rewind(result_set);

        while( !kd_res_end( result_set  )  ) {
                pch = (int)kd_res_itemf( result_set, pos);

                for(t=0; t<inputImages_channels; t++)
                {
                    output_data[outAddress + t] += dists[idx] / (sum_dist) * inputImages_data[pch + t];
                    //printf("%f\n", dists[idx]/sum_dist);
                }
                kd_res_next( result_set );
                idx+=1;
        }


        free(dists);
        kd_res_free(result_set);
      }
    }


  kd_free(kd);
  }
  return 1;
}



int IDWSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput)
{
  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int gradOutput_height = gradOutput->size[1];
  int gradOutput_width = gradOutput->size[2];
  int inputImages_channels = inputImages->size[3];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[1];
  int gradOutput_strideWidth = gradOutput->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[1];
  int gradInputImages_strideWidth = gradInputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[1];
  int gradGrids_strideWidth = gradGrids->stride[2];

  float *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int n;
  n = inputImages_strideBatch * batchsize + inputImages_strideHeight * gradOutput_height + inputImages_strideWidth * gradOutput_width;

  //printf("length %d\n",n);

  int * address_array = malloc(n * sizeof(int));

  int b, yOut, xOut;
  float point_list[gradOutput_height][gradOutput_width][2];
  float grid_list[gradOutput_height][gradOutput_width][2];
  void *kd;




  for(b=0; b < batchsize; b++)
  {
  	kd = kd_create(2);
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        float yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        float xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

	    // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        float yWeightTopLeft, xWeightTopLeft;

        float xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        float ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);

		const int inAddress = inputImages_strideBatch * b + inputImages_strideHeight * yOut + inputImages_strideWidth * xOut;


		const float pt[] = {yf, xf};
        kd_insertf(kd, pt, inAddress);
		address_array[inAddress] = b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth;

        /*const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        float topLeftDotProduct = 0;
        float topRightDotProduct = 0;
        float bottomLeftDotProduct = 0;
        float bottomRightDotProduct = 0;

        float v=0;
        float inTopLeft=0;
        float inTopRight=0;
        float inBottomLeft=0;
        float inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           float gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftIsIn)
           {
              float inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              float inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              float inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              float inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = xf * (inputImages_width-1) / 2;
	*/
      }
    }

	for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
      	float y = (float)(yOut) / (float)(gradOutput_height) * 2 - 1;
        float x = (float)(xOut) / (float)(gradOutput_width) * 2 - 1;
        void *result_set;
        float pt[] = {y, x};
        result_set = kd_nearest_rangef(kd, pt, 3 * 1/(float)(gradOutput_width));
		float pos[2], dist;
        int pch;
        float * dists;
        float sum_dist;
        int npoints = kd_res_size(result_set);
        //printf("%d points found\n",npoints);

        dists = (float *)malloc(npoints * sizeof(float));

        int idx;
        idx = 0;
        sum_dist = 0;
        while( !kd_res_end( result_set  )  ) {
                /* get the data and position of the current result item */
                pch = (int)kd_res_itemf( result_set, pos  );
                    /* compute the distance of the current result from the pt */
                dist = sqrt( dist_sq( pt, pos, 2  )  );
                //printf( "node at (%.3f, %.3f, %.3f) is %.3f away from (%.3f, %.3f, %.3f) and has data=%d\n",
                //                        pos[0], pos[1], pos[2], dist, pt[0], pt[1], pt[2],  pch );
                kd_res_next( result_set );
				
				
               	dists[idx] = 1/(dist + 1e-6);
               	sum_dist += 1/(dist + 1e-6);
				
                idx+=1;
        }
        //backprop happens here
        //
        int t;

        idx = 0;
        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;


        kd_res_rewind(result_set);

        while( !kd_res_end( result_set  )  ) {
                pch = (int)kd_res_itemf( result_set, pos);
                int gradgrid_address = address_array[pch];

                float dotprod = 0.0;

                for(t=0; t<inputImages_channels; t++)
                {
                    //output_data[outAddress + t] += dists[idx] / (sum_dist) * inputImages_data[pch + t];
                    //printf("%f\n", dists[idx]/sum_dist);
					float gradOutValue = gradOutput_data[gradOutputAddress + t];
                    gradInputImages_data[pch + t] += dists[idx] / (sum_dist) * gradOutValue;
                    dotprod += gradOutValue * inputImages_data[pch + t];
                }
								
                gradGrids_data[gradgrid_address] += (pt[0] - pos[0]) * dists[idx] * dists[idx] * dists[idx]* (sum_dist - dists[idx])/(sum_dist * sum_dist) * dotprod;
                gradGrids_data[gradgrid_address+1] += (pt[1] - pos[1]) * dists[idx] * dists[idx] * dists[idx] * (sum_dist - dists[idx])/(sum_dist * sum_dist) * dotprod;
                kd_res_next( result_set );
                idx+=1;
        }

        free(dists);
        kd_res_free(result_set);


	  }

    }


    kd_free(kd);
  }

  free(address_array);
  return 1;
}

static float dist_sq( float *a1, float *a2, int dims  ) {
      float dist_sq = 0, diff;
      while( --dims >= 0  ) {
            diff = (a1[dims] - a2[dims]);
            dist_sq += diff*diff;
      }
        return dist_sq;

}
