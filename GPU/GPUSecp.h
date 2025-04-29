#ifndef GPUSECP
#define GPUSECP

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define NAME_HASH_FOLDER "Hash160"
#define NAME_HASH_BUFFER "merged-sorted-unique-8-byte-hashes"
#define NAME_FILE_OUTPUT "happynation"

//CUDA-specific parameters that determine occupancy and thread-count
//Please read more about them in CUDA docs and adjust according to your GPU specs
#define BLOCKS_PER_GRID 20   //20
#define THREADS_PER_BLOCK 256 //256
#define THREAD_MULT 100 //100 quantas seed cada thread vai calcular até terminar a iteração 


//This is how many hashes are in NAME_HASH_FOLDER, Defined as constant to save one register in device kernel
#define COUNT_INPUT_HASH 1


//CPU stack size in bytes that will be allocated to this program - needs to fit GTable / InputBooks / InputHashes 
#define SIZE_CPU_STACK 1024 * 1024 * 1024

//GPU stack size in bytes that will be allocated to each thread - has complex functionality - please read cuda docs about this
#define SIZE_CUDA_STACK 32768

//---------------------------------------------------------------------------------------------------------------------------
// Don't edit configuration below this line
//---------------------------------------------------------------------------------------------------------------------------

#define SIZE_LONG 8            // Each Long is 8 bytes
#define SIZE_HASH160 20        // Each Hash160 is 20 bytes
#define SIZE_PRIV_KEY 32 	   // Length of the private key that is generated from input seed (in bytes)
#define NUM_GTABLE_CHUNK 16    // Number of GTable chunks that are pre-computed and stored in global memory
#define NUM_GTABLE_VALUE 65536 // Number of GTable values per chunk (all possible states) (2 ^ NUM_GTABLE_CHUNK)
#define SIZE_GTABLE_POINT 32   // Each Point in GTable consists of two 32-byte coordinates (X and Y)
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)
#define COUNT_CUDA_THREADS (BLOCKS_PER_GRID * THREADS_PER_BLOCK)

//Contains the first element index for each chunk
//Pre-computed to save one multiplication
__constant__ int CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

//Contains index multiplied by 8
//Pre-computed to save one multiplication
__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0 + 8,   0 + 16,   0 + 24,   0 + 32,   0 + 40,   0 + 48,   0 + 56,   0 + 64,
   64 + 8,  64 + 16,  64 + 24,  64 + 32,  64 + 40,  64 + 48,  64 + 56,  64 + 64,
  128 + 8, 128 + 16, 128 + 24, 128 + 32, 128 + 40, 128 + 48, 128 + 56, 128 + 64,
  192 + 8, 192 + 16, 192 + 24, 192 + 32, 192 + 40, 192 + 48, 192 + 56, 192 + 64,
  256 + 8, 256 + 16, 256 + 24, 256 + 32, 256 + 40, 256 + 48, 256 + 56, 256 + 64,
  320 + 8, 320 + 16, 320 + 24, 320 + 32, 320 + 40, 320 + 48, 320 + 56, 320 + 64,
  384 + 8, 384 + 16, 384 + 24, 384 + 32, 384 + 40, 384 + 48, 384 + 56, 384 + 64,
  448 + 8, 448 + 16, 448 + 24, 448 + 32, 448 + 40, 448 + 48, 448 + 56, 448 + 64,
};


#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

class GPUSecp
{

public:
	GPUSecp(
		const uint8_t * gTableXCPU,
		const uint8_t * gTableYCPU,
		const uint64_t * inputHashBufferCPU,
		const char* strCPU,          //lucas		
    const int* posicoesCPU,      //lucas
    int totalPosicoesCPUtemp     //lucas
		);

	void doIterationSecp256k1Books(int iteration);
	void doPrintOutput();
	void doFreeMemory();

private:

	char* d_strCPU;								//lucas
  int* d_posicoesCPU;						//lucas
  int d_totalPosicoesCPUtemp;		//lucas



	//GTable buffer containing ~1 million pre-computed points for Secp256k1 point multiplication
	uint8_t * gTableXGPU;
	uint8_t * gTableYGPU;

	//Input buffer that holds merged-sorted-unique-8-byte-hashes in global memory of the GPU device
	uint64_t * inputHashBufferGPU;

	//Output buffer containing result of single iteration
	//If seed created a known Hash160 then outputBufferGPU for that affix will be 1
	uint8_t * outputBufferGPU;
	uint8_t * outputBufferCPU;

	//Output buffer containing result of succesful hash160
	//Each hash160 is 20 bytes long, total size is N * 20 bytes
	uint8_t * outputHashesGPU;
	uint8_t * outputHashesCPU;

	//Output buffer containing private keys that were used in succesful hash160
	//Each private key is 32-byte number that was the output of SHA256
	uint8_t * outputPrivKeysGPU;
	uint8_t * outputPrivKeysCPU;
};



#endif // GPUSecpH
