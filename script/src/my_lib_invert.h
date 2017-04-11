int IDWSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int IDWSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput);

