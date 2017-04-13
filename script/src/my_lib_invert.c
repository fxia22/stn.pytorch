#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>

#define real float

void dot43(real A[4][3], real B[3][3]) {
    int i,j,k;
    for (i = 0; i<3; i++)
    {
            for (j = 0; j<3; j++) {
                B[i][j] = 0;
                for (k = 0; k < 4; k++)
                    B[i][j] += A[k][i] * A[k][j];
        //printf("%f ", B[i][j]);
        }
        //printf("\n");
    }
    //printf("\n");
}


void inv3(real B[3][3], real invB[3][3]) {
    float determinant = +B[0][0]*(B[1][1]*B[2][2]-B[2][1]*B[1][2])
                        -B[0][1]*(B[1][0]*B[2][2]-B[1][2]*B[2][0])
                        +B[0][2]*(B[1][0]*B[2][1]-B[1][1]*B[2][0]);
    float invdet = 1/determinant;
    
    //printf("det %f\n", determinant);
    invB[0][0] =  (B[1][1]*B[2][2]-B[2][1]*B[1][2])*invdet;
    invB[1][0] = -(B[0][1]*B[2][2]-B[0][2]*B[2][1])*invdet;
    invB[2][0] =  (B[0][1]*B[1][2]-B[0][2]*B[1][1])*invdet;
    invB[0][1] = -(B[1][0]*B[2][2]-B[1][2]*B[2][0])*invdet;
    invB[1][1] =  (B[0][0]*B[2][2]-B[0][2]*B[2][0])*invdet;
    invB[2][1] = -(B[0][0]*B[1][2]-B[1][0]*B[0][2])*invdet;
    invB[0][2] =  (B[1][0]*B[2][1]-B[2][0]*B[1][1])*invdet;
    invB[1][2] = -(B[0][0]*B[2][1]-B[2][0]*B[0][1])*invdet;
    invB[2][2] =  (B[0][0]*B[1][1]-B[1][0]*B[0][1])*invdet;
        
}
    

void dot34(real invB[3][3], real A[4][3], real m[3][4]) {
    int i, j, k;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 4; j++){
            m[i][j] = 0;
            for (k = 0; k < 3; k++) {
                m[i][j] += invB[i][k] * A[j][k];
            }
    }
}


void dot41(real m[3][4], real x[4], real alpha[3]) {
    int i,j;
    for (i = 0; i < 3; i++) {
        alpha[i] = 0;
        for (j = 0; j < 4; j++)
            alpha[i] += m[i][j] * x[j];
         //printf("%.2f ", alpha[i]);
    }
    //printf("\n");
}

real min(real * array, int len) {
    real m = array[0];
    int i;
    for (int i = 0; i < len; i++) 
        if (array[i] < m) m = array[i];
    return m;
}

real max(real * array, int len) {
    real m = array[0];
    int i;
    for (int i = 0; i < len; i++) 
        if (array[i] > m) m = array[i];
    return m;
}


void dot21(real im2[2][2], real d[2], real r[2]) {
    int i,j;
    for (i = 0; i < 2; i++) {
        r[i] = 0;
        for (j = 0; j < 2; j++)
            r[i] += im2[i][j] * d[j];
    }
}



void dot22(real m1[2][2], real m2[2][2], real result[2][2]) {
    int i,j,k;
    for (i = 0; i < 2; i++ )
        for (j = 0; j < 2; j++)
        {
            result[i][j] = 0;
            for (k = 0; k < 2; k++)
                result[i][j] += m1[i][k] * m2[k][j];
        }
}


void dot32(real gradalphar[3][2], real gradr[2], real gradalpha[3]) {
    int i,j;
    for (i = 0; i < 3; i++) {
        gradalpha[i] = 0;
        for (j = 0; j < 2; j++) 
            gradalpha[i] += gradalphar[i][j] * gradr[j];
        }
}


void inv2(real m2[2][2], real im2[2][2]) {
   real determinant = m2[0][0] * m2[1][1] - m2[0][1] * m2[1][0];
   //printf("det %.5f\n", determinant);
   im2[0][0] = m2[1][1] / determinant;
   im2[1][1] = m2[0][0] / determinant;
   im2[0][1] = -m2[0][1] / determinant;
   im2[1][0] = -m2[1][0] / determinant;
}

void dot34t(real m[3][4], real alpha[3], real gradx[4]) {
    int i,j;
    for (i = 0; i < 4; i++) {
        gradx[i] = 0;
        for (j = 0; j < 3; j++) 
            gradx[i] += m[j][i] * alpha[j];
    }
}

real abs_real(real num) {
    return (num > 0)?num:-num;
}
    
int InvSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *output)
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


  real *inputImages_data, *output_data, *grids_data, *invgrids_data, *msave_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);
  invgrids_data = THFloatTensor_data(invgrids);
  
  int tradeb, yOut, xOut, k;

  real x[4], y[4], alpha[3], beta[3];
    
  real m2[2][2], im2[2][2];
  
  real minx, miny, minbasex, minbasey;
  real maxx, maxy, maxbasex, maxbasey;
    
  int b;
  
  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height - 1; yOut++)
    {
      for(xOut=0; xOut < output_width - 1; xOut++)
      {
        //read the grid
          

          const int inTopLeftAddress = grids_strideBatch * b + grids_strideHeight * yOut + grids_strideWidth * xOut;
          const int inTopRightAddress = inTopLeftAddress + grids_strideWidth;
          const int inBottomLeftAddress = inTopLeftAddress + grids_strideHeight;
          const int inBottomRightAddress = inBottomLeftAddress + grids_strideWidth;

          
        x[0] = grids_data[inTopLeftAddress + 1];
        x[1] = grids_data[inBottomLeftAddress + 1];
        x[2] = grids_data[inTopRightAddress + 1];
        x[3] = grids_data[inBottomRightAddress + 1];
          
        y[0] = grids_data[inTopLeftAddress];
        y[1] = grids_data[inBottomLeftAddress];
        y[2] = grids_data[inTopRightAddress];
        y[3] = grids_data[inBottomRightAddress];
          
        
        real m[3][4] = {{ 0.7500,    0.2500,    0.2500,   -0.2500},{-0.5000,   -0.5000,    0.5000,    0.5000},{-0.5000,    0.5000,   -0.5000,    0.5000}};  
         
        dot41(m, x, alpha);
        dot41(m, y, beta);
              
        //printf("recon %.4f = %.4f\n", A[0][0] * alpha[0] + A[0][1] * alpha[1] + A[0][2] * alpha[2], x[0]);
        //printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", alpha[0], alpha[1], alpha[2], beta[0], beta[1], beta[2]);    
             
        minx = min(x, 4);
        miny = min(y, 4);
        maxx = max(x, 4);
        maxy = max(y, 4);

         
        int minxcoord = floor((minx + 1) * (inputImages_width - 1)  / 2);
        int maxxcoord = ceil((maxx + 1) * (inputImages_width - 1)  / 2);
          
        int minycoord = floor((miny + 1) * (inputImages_height - 1)  / 2);
        int maxycoord = ceil((maxy + 1) * (inputImages_height - 1)  / 2);
         
        //printf("%d %d %d %d\n", minxcoord, maxxcoord, minycoord, maxycoord);
          
        m2[0][0] = alpha[1];
        m2[0][1] = alpha[2];
        m2[1][0] = beta[1];
        m2[1][1] = beta[2];
        
        inv2(m2, im2);
         
        //printf("%.2f, %.2f \n%.2f, %.2f \n\n", im2[0][0], im2[0][1], im2[1][0], im2[1][1]);
    
        int xcoord, ycoord; 
        if ((maxxcoord - minxcoord) *  (maxycoord - minycoord) < 25)
            for (xcoord = minxcoord;  xcoord < maxxcoord; xcoord ++)
                for (ycoord = minycoord; ycoord < maxycoord; ycoord ++) {
                     
                    real d2[2];
                    real yf = (float)ycoord / (float)(output_height-1) * 2 - 1;
                    real xf = (float)xcoord / (float)(output_width-1) * 2 - 1;
                    
                    d2[0] = xf - alpha[0];
                    d2[1] = yf - beta[0];
                    
                    real r[2];
                    dot21(im2, d2, r); // r[0] x, r[1] y;
            
                    real slack = 1e-3;
                    //printf("%f %f\n", r[0], r[1]);
                    //printf("%.4f = %.4f\n", alpha[0] + alpha[1] * r[0] + alpha[2] * r[1], xf);
                    if ((-slack < r[0]) && (r[0] < 1+slack) &&(-slack < r[1]) && (r[1] < 1 + slack)) {
                        //printf("%.4f, %.4f | %.4f %.4f \n", r[0], r[1], basex[0], basey[0]);
                        int yInTopLeft, xInTopLeft;
                        real yWeightTopLeft, xWeightTopLeft;

                        real xcoord_source = r[0] + xOut;
                        xInTopLeft = floor(xcoord_source);
                        xWeightTopLeft = 1 - (xcoord_source - xInTopLeft);

                        real ycoord_source = r[1] + yOut;
                        yInTopLeft = floor(ycoord_source);
                        yWeightTopLeft = 1 - (ycoord_source - yInTopLeft);
                        
                        const int outAddress = output_strideBatch * b + output_strideHeight * ycoord + output_strideWidth * xcoord;
                        const int outGridAddress = grids_strideBatch * b + grids_strideHeight * ycoord + grids_strideWidth * xcoord;
                        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
                        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
                        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
                        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;
                                                                   
                        real v=0;
                        real inTopLeft=0;
                        real inTopRight=0;
                        real inBottomLeft=0;
                        real inBottomRight=0;
                        
                        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
                        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
                        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
                        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

                        bool outIsIn =  xcoord >= 0 && xcoord <= inputImages_width-1 && ycoord >= 0 && ycoord <= inputImages_height-1;
                        
                        int t;          
                        
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

                           if (outIsIn) output_data[outAddress + t] = v;
                        }
                        
                        if (outIsIn) invgrids_data[outGridAddress] = (float)yOut;
                        if (outIsIn) invgrids_data[outGridAddress+1] = (float)xOut; // x - [+1], y - [0]
                        
               
                    } 
        }

      }
    }
  }

  return 1;
}

int InvSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *invgrids, THFloatTensor *gradInputImages, THFloatTensor *gradGrids, THFloatTensor *gradOutput)
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
   


  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data, *invgrids_data;
  
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  invgrids_data = THFloatTensor_data(invgrids);
    
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;
   
  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
          const int Address = gradGrids_strideBatch * b + gradGrids_strideHeight * yOut + gradGrids_strideWidth * xOut;
          gradGrids_data[Address] = 0;
          gradGrids_data[Address + 1] = 0;
          
      }
    }
  }
    
   for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
          const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;
          const int invgridAddress = grids_strideBatch * b + grids_strideHeight * yOut + grids_strideWidth * xOut;
          
          real r[2], gradr[2];
          
          int xSource, ySource;
          
          xSource = (int)invgrids_data[invgridAddress + 1];
          ySource = (int)invgrids_data[invgridAddress];
          
          //printf("%d %d\n", xSource ,ySource);
          
          const int gridinTopLeftAddress = grids_strideBatch * b + grids_strideHeight * ySource + grids_strideWidth * xSource;
          const int gridinTopRightAddress = gridinTopLeftAddress + grids_strideWidth;
          const int gridinBottomLeftAddress = gridinTopLeftAddress + grids_strideHeight;
          const int gridinBottomRightAddress = gridinBottomLeftAddress + grids_strideWidth;
          
          int i,j;
          
          real m[3][4] = {{ 0.7500,    0.2500,    0.2500,   -0.2500},{-0.5000,   -0.5000,    0.5000,    0.5000},{-0.5000,    0.5000,   -0.5000,    0.5000}};  
          
          real gradalpha[3], gradbeta[3], alpha[3], beta[3];
         
          real x[4], y[4];
          x[0] = grids_data[gridinTopLeftAddress + 1];
          x[1] = grids_data[gridinBottomLeftAddress + 1];
          x[2] = grids_data[gridinTopRightAddress + 1];
          x[3] = grids_data[gridinBottomRightAddress + 1];

          y[0] = grids_data[gridinTopLeftAddress];
          y[1] = grids_data[gridinBottomLeftAddress];
          y[2] = grids_data[gridinTopRightAddress];
          y[3] = grids_data[gridinBottomRightAddress];
          
          dot41(m, x, alpha);
          dot41(m, y, beta);
          real target_yf, target_xf;
          target_yf = (float)yOut / (float)(inputImages_height - 1) * 2 - 1;
          target_xf = (float)xOut / (float)(inputImages_width - 1) * 2 - 1;
          
          real m2[2][2], im2[2][2];
          m2[0][0] = alpha[1];
          m2[0][1] = alpha[2];
          m2[1][0] = beta[1];
          m2[1][1] = beta[2];
          inv2(m2, im2);
          
          real d2[2];
          d2[0] = target_xf - alpha[0];
          d2[1] = target_yf - beta[0];
          
          real r2[2];
          
          dot21(im2, d2, r2);
          
          
          if ((xSource != 0) || (ySource != 0)) {
               //printf("%d %d %.8f %.8f\n", xSource ,ySource, r2[0], r2[1]);
              
                // get the weights for interpolation
                int yInTopLeft, xInTopLeft;
                real yWeightTopLeft, xWeightTopLeft;
                real xgrad,ygrad;
                
                xInTopLeft = xSource;
                xWeightTopLeft = r[0];

                yInTopLeft = ySource;
                yWeightTopLeft = r[1];

                const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
                const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
                const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
                const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

                const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
                const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
                const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
                const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

                const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

                real topLeftDotProduct = 0;
                real topRightDotProduct = 0;
                real bottomLeftDotProduct = 0;
                real bottomRightDotProduct = 0;

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

                for(t=0; t<inputImages_channels; t++)
                {
                   real gradOutValue = gradOutput_data[gradOutputAddress + t];
                   if(topLeftIsIn)
                   {
                      real inTopLeft = inputImages_data[inTopLeftAddress + t];
                      topLeftDotProduct += inTopLeft * gradOutValue;
                      if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
                   }

                   if(topRightIsIn)
                   {
                      real inTopRight = inputImages_data[inTopRightAddress + t];
                      topRightDotProduct += inTopRight * gradOutValue;
                      if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
                   }

                   if(bottomLeftIsIn)
                   {
                      real inBottomLeft = inputImages_data[inBottomLeftAddress + t];
                      bottomLeftDotProduct += inBottomLeft * gradOutValue;
                      if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
                   }

                   if(bottomRightIsIn)
                   {
                      real inBottomRight = inputImages_data[inBottomRightAddress + t];
                      bottomRightDotProduct += inBottomRight * gradOutValue;
                      if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
                   }
                }

                ygrad = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
                xgrad = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

                
                 //printf("%f %f\n", xgrad, ygrad);
              
              gradr[0] = xgrad;
              gradr[1] = ygrad;
              
              
              real grad_alpha_r[3][2], grad_beta_r[3][2], gradalpha[3], gradbeta[3];

              m2[0][0] = alpha[1];
              m2[0][1] = alpha[2];
              m2[1][0] = beta[1];
              m2[1][1] = beta[2];
              inv2(m2, im2);

              d2[0] = target_xf - alpha[0];
              d2[1] = target_yf - beta[0];


              real i00[2][2] = {{1,0},{0,0}};
              real temp[2][2];
              real temp2[2][2], tempgrad[2];

              dot22(im2, i00, temp);
              dot22(temp, im2, temp2);
              dot21(temp2, d2, tempgrad);
              grad_alpha_r[1][0] = -tempgrad[0];
              grad_alpha_r[1][1] = -tempgrad[1];


              real i01[2][2] = {{0,1},{0,0}};
              dot22(im2, i01, temp);
              dot22(temp, im2, temp2);
              dot21(temp2, d2, tempgrad);
              grad_alpha_r[2][0] = -tempgrad[0];
              grad_alpha_r[2][1] = -tempgrad[1];


              real i10[2][2] = {{0,0},{1,0}};
              dot22(im2, i10, temp);
              dot22(temp, im2, temp2);
              dot21(temp2, d2, tempgrad);
              grad_beta_r[1][0] = -tempgrad[0];
              grad_beta_r[1][1] = -tempgrad[1];


              real i11[2][2] = {{0,0},{0,1}};
              dot22(im2, i11, temp);
              dot22(temp, im2, temp2);
              dot21(temp2, d2, tempgrad);
              grad_beta_r[2][0] = -tempgrad[0];
              grad_beta_r[2][1] = -tempgrad[1];

              real j0[2] = {1,0};
              dot21(im2, j0, tempgrad);
              grad_alpha_r[0][0] = -tempgrad[0];
              grad_alpha_r[0][1] = -tempgrad[1];

              real j1[2] = {0,1};
              dot21(im2, j1, tempgrad);
              grad_beta_r[0][0] = -tempgrad[0];
              grad_beta_r[0][1] = -tempgrad[1];


              dot32(grad_beta_r, gradr, gradbeta);
              dot32(grad_alpha_r, gradr, gradalpha);
              
              //printf("%.3f %.3f %.3f\n", gradbeta[0], gradbeta[1], gradbeta[2]);
              
              real gradx[4], grady[4];
              dot34t(m, gradalpha, gradx);
              dot34t(m, gradbeta, grady);


              //printf("x %.3f %.3f %.3f %.3f\n", gradx[0], gradx[1], gradx[2], gradx[3]);
              //printf("y %.3f %.3f %.3f %.3f\n", grady[0], grady[1], grady[2], grady[3]);
              
              gradGrids_data[gridinTopLeftAddress] += grady[0];
              gradGrids_data[gridinTopLeftAddress + 1] += gradx[0];
              
              gradGrids_data[gridinBottomLeftAddress] += grady[1];
              gradGrids_data[gridinBottomLeftAddress + 1] += gradx[1];
              
              gradGrids_data[gridinTopRightAddress] += grady[2];
              gradGrids_data[gridinTopRightAddress + 1] += gradx[2];
              
              gradGrids_data[gridinBottomRightAddress] += grady[3];
              gradGrids_data[gridinBottomRightAddress + 1] += gradx[3];
              
          
          }
           
      }
    }
   }

  return 1;
}
