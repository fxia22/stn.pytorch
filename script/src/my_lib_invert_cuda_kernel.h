#ifdef __cplusplus
extern "C" {
#endif

int InvSamplerBHWD_updateOutput_cuda_kernel(
	  int batchsize 			,//= inputImages->size[0];
	  int inputImages_height 	,//= inputImages->size[1];
	  int inputImages_width	 	,//= inputImages->size[2];
	  int output_height 		,//= output->size[1];
	  int output_width 			,//= output->size[2];
	  int inputImages_channels 	,//= inputImages->size[3];
	  int output_strideBatch 	,//= output->stride[0];
	  int output_strideHeight 	,//= output->stride[1];
	  int output_strideWidth 	,//= output->stride[2];
	  int depth_strideBatch 	,//= depth_map->stride[0];
	  int depth_strideHeight 	,//= depth_map->stride[1];
	  int depth_strideWidth 	,//= depth_map->stride[2];
	  int inputImages_strideBatch 	,//= inputImages->stride[0];
	  int inputImages_strideHeight 	,//= inputImages->stride[1];
	  int inputImages_strideWidth 	,//= inputImages->stride[2];
	  int grids_strideBatch 	,//= grids->stride[0];
	  int grids_strideHeight 	,//= grids->stride[1];
	  int grids_strideWidth 	,//= grids->stride[2];
	  float *inputImages_data, 
	  float *output_data, 
	  float *grids_data, 
	  float *invgrids_data, 
	  float *depth_data,
	  float *target_depth_data,
	  cudaStream_t stream);


int InvSamplerBHWD_updateGradInput_cuda_kernel(
    int batchsize 					,//= inputImages->size[0];
    int inputImages_height 			,//= inputImages->size[1];
    int inputImages_width 			,//= inputImages->size[2];
    int gradOutput_height 			,//= gradOutput->size[1];
    int gradOutput_width 				,//= gradOutput->size[2];
    int inputImages_channels 			,//= inputImages->size[3];
    int gradOutput_strideBatch 		,//= gradOutput->stride[0];
    int gradOutput_strideHeight 		,//= gradOutput->stride[1];
    int gradOutput_strideWidth 		,//= gradOutput->stride[2];
    int inputImages_strideBatch 		,//= inputImages->stride[0];
    int inputImages_strideHeight 		,//= inputImages->stride[1];
    int inputImages_strideWidth 		,//= inputImages->stride[2];
    int gradInputImages_strideBatch 	,//= gradInputImages->stride[0];
    int gradInputImages_strideHeight 	,//= gradInputImages->stride[1];
    int gradInputImages_strideWidth 	,//= gradInputImages->stride[2];
    int grids_strideBatch 			,//= grids->stride[0];
    int grids_strideHeight 			,//= grids->stride[1];
    int grids_strideWidth 			,//= grids->stride[2];
    int gradGrids_strideBatch 		,//= gradGrids->stride[0];
    int gradGrids_strideHeight 		,//= gradGrids->stride[1];
    int gradGrids_strideWidth 		,//= gradGrids->stride[2];
    float *inputImages_data, 
	float *gradOutput_data,  
	float *grids_data,  
	float *gradGrids_data, 
	float *gradInputImages_data, 
	float *invgrids_data,
	cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
