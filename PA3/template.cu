
#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int width, int numBCol) {
  //@@ Insert code to implement matrix multiplication here
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if((Row < width) && (Col << width)){
    float Pvalue = 0;
    for(int k = 0; k < width; k++){
      Pvalue += A[Row * width + k] * B[k * numBCol + Col];
    }
    C[Row * numBCol + Col] = Pvalue;
  }

}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A3
  int numAColumns; // number of columns in the matrix A4
  int numBRows;    // number of rows in the matrix B4
  int numBColumns; // number of columns in the matrix B5
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int width;
  int BLOCK_WIDTH = 4;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Set width of the matrix
  width = numARows;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc((numCRows * numCColumns) * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, (numARows * numAColumns) * sizeof(float));
  cudaMalloc((void **) &deviceB, (numBRows * numBColumns) * sizeof(float));
  cudaMalloc((void **) &deviceC, (numCRows * numCColumns) * sizeof(float));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, (numARows * numAColumns) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, (numBRows * numBColumns) * sizeof(float), cudaMemcpyHostToDevice);
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // dim3 dimGrid(ceil((1.0*numCRows)/BLOCK_WIDTH), 
	// 		 ceil((1.0*numCRows)/BLOCK_WIDTH), 1);
  //dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 dimGrid(ceil((1.0*width)/BLOCK_WIDTH),ceil((1.0*width)/BLOCK_WIDTH),1);
  dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, width, numBColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
