#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <pthread.h>
#include <fstream>
#include "GPU/GPUSecp.h"
#include "CPU/SECP256k1.h"
#include "CPU/HashMerge.cpp"
#include <sys/resource.h>
#include <chrono>

#include <cmath> //pow

long getFileContent(std::string fileName, std::vector<std::string> &vecOfStrs) {
	long totalSizeBytes = 0;

	std::ifstream in(fileName.c_str());
	if (!in)
	{
		//std::cerr << "Can not open the File : " << fileName << std::endl;
		return 0;
	}
	std::string str;
	while (std::getline(in, str))
	{
		vecOfStrs.push_back(str);
		totalSizeBytes += str.size();
	}
	
	in.close();
	return totalSizeBytes;
}

bool lerChaveDeArquivo(const char* nomeArquivo, char* strCPU, uint8_t* privKeycpu) {
    std::ifstream inputFile(nomeArquivo);
    if (!inputFile) {
        std::cerr << "Erro ao abrir o arquivo " << nomeArquivo << std::endl;
        return false;
    }

    std::string line;
    if (!std::getline(inputFile, line)) {
        std::cerr << "Erro ao ler linha do arquivo " << nomeArquivo << std::endl;
        return false;
    }

    if (line.length() != 64) {
        std::cerr << "Erro: a linha do arquivo deve conter exatamente 64 caracteres." << std::endl;
        return false;
    }

    strncpy(strCPU, line.c_str(), 64);
    strCPU[64] = '\0'; // garante término nulo
    
    // Converte a string hex para uint8_t[32] em little-endian
	for (int i = 0; i < 32; i++) {
    	// Lê os caracteres hex *do fim para o início* (inverte a ordem dos bytes)
    	int posChar = (31 - i) * 2;  // Começa do último par de caracteres
    	char highChar = strCPU[posChar];
    	char lowChar = strCPU[posChar + 1];

    	// Converte para nibbles e combina em um byte
    	uint8_t highNibble = (highChar >= '0' && highChar <= '9') ? highChar - '0' :
                        (highChar >= 'a' && highChar <= 'f') ? highChar - 'a' + 10 : 0;
    	uint8_t lowNibble = (lowChar >= '0' && lowChar <= '9') ? lowChar - '0' :
                       (lowChar >= 'a' && lowChar <= 'f') ? lowChar - 'a' + 10 : 0;

    	privKeycpu[i] = (highNibble << 4) | lowNibble;
	}
    return true;
}

void loadInputHash(uint64_t *inputHashBufferCPU) {
	std::cout << "Loading hash buffer from file: " << NAME_HASH_BUFFER << std::endl;

	FILE *fileSortedHash = fopen(NAME_HASH_BUFFER, "rb");
	if (fileSortedHash == NULL)
	{
		printf("Error: not able to open input file: %s\n", NAME_HASH_BUFFER);
		exit(1);
	}

	fseek(fileSortedHash, 0, SEEK_END);
	long hashBufferSizeBytes = ftell(fileSortedHash);
	long hashCount = hashBufferSizeBytes / SIZE_LONG;
	rewind(fileSortedHash);

	if (hashCount != COUNT_INPUT_HASH) {
		printf("ERROR - Constant COUNT_INPUT_HASH is %d, but the actual hashCount is %lu \n", COUNT_INPUT_HASH, hashCount);
		exit(-1);
	}

	size_t size = fread(inputHashBufferCPU, 1, hashBufferSizeBytes, fileSortedHash);
	fclose(fileSortedHash);

	std::cout << "loadInputHash " << NAME_HASH_BUFFER << " finished!" << std::endl;
	std::cout << "hashCount: " << hashCount << ", hashBufferSizeBytes: " << hashBufferSizeBytes << std::endl;
}

void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
	std::cout << "loadGTable started" << std::endl;

	Secp256K1 *secp = new Secp256K1();
	secp->Init();

	for (int i = 0; i < NUM_GTABLE_CHUNK; i++)
	{
		for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++)
		{
			int element = (i * NUM_GTABLE_VALUE) + j;
			Point p = secp->GTable[element];
			for (int b = 0; b < 32; b++) {
				gTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
				gTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
			}
		}
	}

	std::cout << "loadGTable finished!" << std::endl;
}

void salvarPosicoes(const char *strCPU, int *posicoesCPU, int *totalPosicoesCPU) {
    *totalPosicoesCPU = 0;
    for (int i = 0; strCPU[i] != '\0'; i++) {
        if (strCPU[i] == 'x') {
            posicoesCPU[*totalPosicoesCPU] = i;
            (*totalPosicoesCPU)++;
        }
    }
    printf("Total de posições 'x' encontradas:%d\n", *totalPosicoesCPU);
    printf("X nas posições no array:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]); 
    }
    printf("      ");
    printf("X nas posições na chave:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]+1); 
    }
    printf("\n");
}



void startSecp256k1ModeBooks(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU) {

	printf("Modo HEX \n");

	char strCPU[65] = {0};
	uint8_t privKeycpu[SIZE_PRIV_KEY];
		if (!lerChaveDeArquivo("chave.txt", strCPU, privKeycpu)) {
    		exit(1);
		}printf("Chave Parcial char: %s    \n", strCPU);



	printf("privkey cpu hex: ");
	for (int i = 0; i < 32; ++i) {
    	// cast para unsigned para evitar problemas com promotions
    	printf("%02X", (unsigned)privKeycpu[i]);
	}printf("\n");

	int posicoesCPU[65];
	int totalPosicoesCPUtemp;

	salvarPosicoes(strCPU, posicoesCPU, &totalPosicoesCPUtemp);

	GPUSecp *gpuSecp = new GPUSecp(
		gTableXCPU,
		gTableYCPU,
		inputHashBufferCPU,
		privKeycpu,              
    	posicoesCPU,         
    	totalPosicoesCPUtemp
	);

	long timeTotal = 0;
	long totalCount = (COUNT_CUDA_THREADS);

	long long possibilidades = pow(16, totalPosicoesCPUtemp)-1;
	printf("possibilidades %lld \n",possibilidades);

	int itercount = COUNT_CUDA_THREADS * THREAD_MULT;
	int maxIteration = possibilidades / itercount;
	printf("cada iteração resulta em %d tentativas, resultando em no maximo de %d iterações \n", itercount, maxIteration);

	for (int iter = 0; iter < maxIteration+1; iter++) {
		const auto clockIter1 = std::chrono::system_clock::now();
		gpuSecp->doIterationSecp256k1Books(iter);
		const auto clockIter2 = std::chrono::system_clock::now();
		gpuSecp->doPrintOutput();

		long timeIter1 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter1.time_since_epoch()).count();
		long timeIter2 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2.time_since_epoch()).count();
		long iterationDuration = (timeIter2 - timeIter1);
		timeTotal += iterationDuration;

		totalCount = (itercount * (iter+1));
		printf("Cambuca Iteration: %d, time: %ld \r", iter, iterationDuration);

		int restantes = maxIteration - iter;
		long etaMillis = iterationDuration * restantes;

		int etaSeconds = etaMillis / 1000;
		int etaMinutes = etaSeconds / 60;
		int etaRemSeconds = etaSeconds % 60;

		int progresso = (int)((iter + 1) * 100.0 / (maxIteration + 1));
		static int lastPrintedPercent = -1;

		if (progresso > lastPrintedPercent) {
			lastPrintedPercent = progresso;
			printf("Iteração %d/%d, Tempo: %ld ms, Progresso: %d%%, ETA: %d min %d s\r",iter + 1, maxIteration + 1, iterationDuration, progresso, etaMinutes, etaRemSeconds);
			fflush(stdout);
		}		
		//printf("Iteração %d/%d, Tempo: %ld ms, Progresso: %d%%, ETA: %d min %d s\r",iter + 1, maxIteration + 1, iterationDuration, progresso, etaMinutes, etaRemSeconds);

	}

	printf("Finished in %ld milliseconds (%.2f seconds)\n", timeTotal, timeTotal / 1000.0);

	printf("Total Seed Count: %lld \n", possibilidades);

	printf("Seeds Per Second: %0.2lf Mkeys\n", possibilidades / (double)(timeTotal * 1000));
}

void increaseStackSizeCPU() {
	const rlim_t cpuStackSize = SIZE_CPU_STACK;
	struct rlimit rl;
	int result;

	printf("Increasing Stack Size to %lu \n", cpuStackSize);

	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < cpuStackSize)
		{
			rl.rlim_cur = cpuStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

int main(int argc, char **argv) {
	printf("iniciando a cambuca \n");

	increaseStackSizeCPU();

	mergeHashes(NAME_HASH_FOLDER, NAME_HASH_BUFFER);

	uint8_t* gTableXCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
	uint8_t* gTableYCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];

	loadGTable(gTableXCPU, gTableYCPU);

	uint64_t* inputHashBufferCPU = new uint64_t[COUNT_INPUT_HASH];

	loadInputHash(inputHashBufferCPU);

	startSecp256k1ModeBooks(gTableXCPU, gTableYCPU, inputHashBufferCPU);
	
	free(gTableXCPU);
	free(gTableYCPU);
	free(inputHashBufferCPU);

	printf("	Complete \n");
	return 0;
}
