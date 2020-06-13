/*
Header file including necessary nvml headers.
*/

#ifndef INCLNVML
#define INCLNVML

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include "Rapl.h"

#define SAMPLE_MS  10
#define COOLDOWN_MS  100


// GPU power measure functions
void GPUPowerBegin(const char *alg);
void GPUPowerEnd();

// CPU power measure functions
void CPUPowerBegin(const char *alg);
void CPUPowerEnd();

// pthread functions
void *GPUpowerPollingFunc(void *ptr);
void *CPUpowerPollingFunc(void *ptr);
int getNVMLError(nvmlReturn_t resultToCheck);

#endif
