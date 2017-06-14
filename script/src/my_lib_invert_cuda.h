int InvSamplerBHWD_updateOutput_cuda(THCudaTensor *inputImages,
        							 THCudaTensor *grids, 
									 THCudaTensor *invgrids,
        							 THCudaTensor *output, 
									 THCudaTensor *depth_map,
									 THCudaTensor *target_depth_map,
									 int * device);

int InvSamplerBHWD_updateGradInput_cuda(THCudaTensor *inputImages, 
										THCudaTensor *grids, 
										THCudaTensor *invgrids, 
										THCudaTensor *gradInputImages, 
										THCudaTensor *gradGrids, 
										THCudaTensor *gradOutput,
										int * device);

//int InvSamplerBHWD_updateGradInput_num(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, THFloatTensor *msave);

