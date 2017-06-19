#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "my_lib_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int InvSamplerBHWD_updateOutput_cuda(THCudaTensor *inputImages,
        							 THCudaTensor *grids, 
									 THCudaTensor *invgrids,
        							 THCudaTensor *output, 
									 THCudaTensor *depth_map,
								     THCudaTensor *target_depth_map,
									 int * device)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  cudaSetDevice(device[0]);
  int success = 0;
  
  success =  InvSamplerBHWD_updateOutput_cuda_kernel(
	  THCudaTensor_size(state, inputImages, 0), //int batchsize 			,//= inputImages->size[0];
	  THCudaTensor_size(state, inputImages, 1),	//int inputImages_height 	,//= inputImages->size[1];
	  THCudaTensor_size(state, inputImages, 2),	//int inputImages_width	 	,//= inputImages->size[2];
	  THCudaTensor_size(state, output, 1),		//int output_height 		,//= output->size[1];
	  THCudaTensor_size(state, output, 2),		//int output_width 			,//= output->size[2];
	  THCudaTensor_size(state, inputImages, 3),	//int inputImages_channels 	,//= inputImages->size[3];
	  THCudaTensor_stride(state, output, 0),//int output_strideBatch 	,//= output->stride[0];
	  THCudaTensor_stride(state, output, 1),//int output_strideHeight 	,//= output->stride[1];
	  THCudaTensor_stride(state, output, 2),//int output_strideWidth 	,//= output->stride[2];
	  THCudaTensor_stride(state, depth_map, 0),//int depth_strideBatch 	,//= depth_map->stride[0];
	  THCudaTensor_stride(state, depth_map, 1),//int depth_strideHeight 	,//= depth_map->stride[1];
	  THCudaTensor_stride(state, depth_map, 2),//int depth_strideWidth 	,//= depth_map->stride[2];
	  THCudaTensor_stride(state, inputImages, 0),//int inputImages_strideBatch 	,//= inputImages->stride[0];
	  THCudaTensor_stride(state, inputImages, 1),//int inputImages_strideHeight 	,//= inputImages->stride[1];
	  THCudaTensor_stride(state, inputImages, 2),//int inputImages_strideWidth 	,//= inputImages->stride[2];
	  THCudaTensor_stride(state, grids, 0),//int grids_strideBatch 	,//= grids->stride[0];
	  THCudaTensor_stride(state, grids, 1),//int grids_strideHeight 	,//= grids->stride[1];
	  THCudaTensor_stride(state, grids, 2),//int grids_strideWidth 	,//= grids->stride[2];
	  THCudaTensor_data(state, inputImages),//float *inputImages_data, 
	  THCudaTensor_data(state, output),//float *output_data, 
	  THCudaTensor_data(state, grids),//float *grids_data, 
	  THCudaTensor_data(state, invgrids),//float *invgrids_data, 
	  THCudaTensor_data(state, depth_map),//float *depth_data,
	  THCudaTensor_data(state, target_depth_map),//float *target_depth_data,
	  THCState_getCurrentStream(state)); //cudaStream_t stream
  
  
  
//  success = BilinearSamplerBHWD_updateOutput_cuda_kernel(output->size[2],
//                                               output->size[1],
//                                               output->size[0],
//                                               THCudaTensor_size(state, inputImages, 3),
//                                               THCudaTensor_size(state, inputImages, 1),
//                                               THCudaTensor_size(state, inputImages, 2),
//                                               THCudaTensor_size(state, output, 2),
//                                               THCudaTensor_data(state, inputImages),
//                                               THCudaTensor_stride(state, inputImages, 0),
//                                               THCudaTensor_stride(state, inputImages, 3),
//                                               THCudaTensor_stride(state, inputImages, 1),
//                                               THCudaTensor_stride(state, inputImages, 2),
//                                               THCudaTensor_data(state, grids),
//                                               THCudaTensor_stride(state, grids, 0),
//                                               THCudaTensor_stride(state, grids, 3),
//                                               THCudaTensor_stride(state, grids, 1),
//                                               THCudaTensor_stride(state, grids, 2),
//                                               THCudaTensor_data(state, output),
//                                               THCudaTensor_stride(state, output, 0),
//                                               THCudaTensor_stride(state, output, 3),
//                                               THCudaTensor_stride(state, output, 1),
//                                               THCudaTensor_stride(state, output, 2),
//                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int InvSamplerBHWD_updateGradInput_cuda(THCudaTensor *inputImages, 
										THCudaTensor *grids, 
										THCudaTensor *invgrids, 
										THCudaTensor *gradInputImages, 
										THCudaTensor *gradGrids, 
										THCudaTensor *gradOutput,
										int * device)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  cudaSetDevice(device[0]);
  int success = 0;
  
  success = InvSamplerBHWD_updateGradInput_cuda_kernel(
    THCudaTensor_size(state, inputImages, 0),//int batchsize 					,//= inputImages->size[0];
    THCudaTensor_size(state, inputImages, 1),//int inputImages_height 			,//= inputImages->size[1];
    THCudaTensor_size(state, inputImages, 2),//int inputImages_width 			,//= inputImages->size[2];
    THCudaTensor_size(state, gradOutput, 1),//int gradOutput_height 			,//= gradOutput->size[1];
    THCudaTensor_size(state, gradOutput, 2),//int gradOutput_width 				,//= gradOutput->size[2];
   	THCudaTensor_size(state, inputImages, 3), //int inputImages_channels 			,//= inputImages->size[3];
    THCudaTensor_stride(state, gradOutput, 0),//int gradOutput_strideBatch 		,//= gradOutput->stride[0];
    THCudaTensor_stride(state, gradOutput, 1),//int gradOutput_strideHeight 		,//= gradOutput->stride[1];
    THCudaTensor_stride(state, gradOutput, 2),//int gradOutput_strideWidth 		,//= gradOutput->stride[2];
    THCudaTensor_stride(state, inputImages, 0),//int inputImages_strideBatch 		,//= inputImages->stride[0];
    THCudaTensor_stride(state, inputImages, 1),//int inputImages_strideHeight 		,//= inputImages->stride[1];
    THCudaTensor_stride(state, inputImages, 2),//int inputImages_strideWidth 		,//= inputImages->stride[2];
    THCudaTensor_stride(state, gradInputImages, 0),//int gradInputImages_strideBatch 	,//= gradInputImages->stride[0];
    THCudaTensor_stride(state, gradInputImages, 1),//int gradInputImages_strideHeight 	,//= gradInputImages->stride[1];
    THCudaTensor_stride(state, gradInputImages, 2),//int gradInputImages_strideWidth 	,//= gradInputImages->stride[2];
    THCudaTensor_stride(state, grids, 0),//int grids_strideBatch 			,//= grids->stride[0];
    THCudaTensor_stride(state, grids, 1),//int grids_strideHeight 			,//= grids->stride[1];
    THCudaTensor_stride(state, grids, 2),//int grids_strideWidth 			,//= grids->stride[2];
    THCudaTensor_stride(state, gradGrids, 0),//int gradGrids_strideBatch 		,//= gradGrids->stride[0];
    THCudaTensor_stride(state, gradGrids, 1),//int gradGrids_strideHeight 		,//= gradGrids->stride[1];
    THCudaTensor_stride(state, gradGrids, 2),//int gradGrids_strideWidth 		,//= gradGrids->stride[2];
    THCudaTensor_data(state, inputImages),//float *inputImages_data, 
	THCudaTensor_data(state, gradOutput),//float *gradOutput_data,  
	THCudaTensor_data(state, grids),//float *grids_data,  
	THCudaTensor_data(state, gradGrids),//float *gradGrids_data, 
	THCudaTensor_data(state, gradInputImages),//float *gradInputImages_data, 
	THCudaTensor_data(state, invgrids),//float *invgrids_data,
	THCState_getCurrentStream(state));//cudaStream_t stream
  
//  success = BilinearSamplerBHWD_updateGradInput_cuda_kernel(gradOutput->size[2],
//                                                  gradOutput->size[1],
//                                                  gradOutput->size[0],
//                                                  THCudaTensor_size(state, inputImages, 3),
//                                                  THCudaTensor_size(state, inputImages, 1),
//                                                  THCudaTensor_size(state, inputImages, 2),
//                                                  THCudaTensor_size(state, gradOutput, 2),
//                                                  THCudaTensor_data(state, inputImages),
//                                                  THCudaTensor_stride(state, inputImages, 0),
//                                                  THCudaTensor_stride(state, inputImages, 3),
//                                                  THCudaTensor_stride(state, inputImages, 1),
//                                                  THCudaTensor_stride(state, inputImages, 2),
//                                                  THCudaTensor_data(state, grids),
//                                                  THCudaTensor_stride(state, grids, 0),
//                                                  THCudaTensor_stride(state, grids, 3),
//                                                  THCudaTensor_stride(state, grids, 1),
//                                                  THCudaTensor_stride(state, grids, 2),
//                                                  THCudaTensor_data(state, gradInputImages),
//                                                  THCudaTensor_stride(state, gradInputImages, 0),
//                                                  THCudaTensor_stride(state, gradInputImages, 3),
//                                                  THCudaTensor_stride(state, gradInputImages, 1),
//                                                  THCudaTensor_stride(state, gradInputImages, 2),
//                                                  THCudaTensor_data(state, gradGrids),
//                                                  THCudaTensor_stride(state, gradGrids, 0),
//                                                  THCudaTensor_stride(state, gradGrids, 3),
//                                                  THCudaTensor_stride(state, gradGrids, 1),
//                                                  THCudaTensor_stride(state, gradGrids, 2),
//                                                  THCudaTensor_data(state, gradOutput),
//                                                  THCudaTensor_stride(state, gradOutput, 0),
//                                                  THCudaTensor_stride(state, gradOutput, 3),
//                                                  THCudaTensor_stride(state, gradOutput, 1),
//                                                  THCudaTensor_stride(state, gradOutput, 2),
//                                                  THCState_getCurrentStream(state));
//
  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}



