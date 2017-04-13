int InvSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *output, THFloatTensor *depth_map);

int InvSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput);

//int InvSamplerBHWD_updateGradInput_num(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput, THFloatTensor *msave);

