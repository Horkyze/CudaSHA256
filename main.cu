#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.cuh"


__global__ void sha256_cuda(JOB ** jobs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
	sha256_final(&ctx, jobs[i]->digest);
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n)
{
	sha256_cuda <<< 1, n >>> (jobs);
}

void print_jobs(JOB ** jobs, int n) {
	printf("@ %p jobs  \n", jobs);
	for (int i = 0; i < n; i++)
	{
		printf("@ %p JOB[%i] \n", jobs[i], i);
		printf("\t @ 0x%p data = %x \n", jobs[i]->data, jobs[i]->data[0]);
		printf("\t @ 0x%p size = %llu \n", &(jobs[i]->size), jobs[i]->size);
		printf("\t @ 0x%p digest = %s \n------\n", jobs[i]->digest, print_sha(jobs[i]->digest));
	}
}

JOB * JOB_init() {
	JOB * j;
	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));
	checkCudaErrors(cudaMallocManaged(&(j->data), 2));
	cudaMemset(j->data, 0xff, 2);
	j->size = 2;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	return j;
}

int main()
{
	// parse input

	// number of jobs
	int n = 3;


	// create JOB array
	JOB ** jobs;
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
	//jobs = (JOB **) malloc(n * sizeof(JOB *));
	int i = 0;
	for (i = 0; i < n; i++)
	{
		jobs[i] = JOB_init();
	}
	print_jobs(jobs, n);

	pre_sha256();
	runJobs(jobs, n);

	cudaDeviceSynchronize();

	print_jobs(jobs, n);

	cudaDeviceReset();
    return 0;
}
