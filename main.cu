#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sha256.cuh"


__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n)
{
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}

void print_jobs(JOB ** jobs, int n) {
	printf("@ %p jobs  \n", jobs);
	for (int i = 0; i < n; i++)
	{
		printf("@ %p JOB[%i] \n", jobs[i], i);
		printf("\t @ 0x%p data = %x \n", jobs[i]->data, (jobs[i]->data == 0)? 0 : jobs[i]->data[0]);
		printf("\t @ 0x%p size = %llu \n", &(jobs[i]->size), jobs[i]->size);
		printf("\t @ 0x%p digest = %s \n------\n", jobs[i]->digest, print_sha(jobs[i]->digest));
	}
}

JOB * JOB_init(BYTE * data, long size) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	return j;
}

void print_usage(){
	printf("/.CuadaSHA256 <file>");
}


BYTE * get_file_data(char * fname, unsigned long * size){
	FILE * f = 0;
	BYTE * buffer = 0;
	unsigned long fsize = 0;

	f = fopen(fname, "rb");
	if (!f){
		fprintf(stderr, "Unable to open %s\n", fname);
		return 0;
	}
	fflush(f);

	if (fseek(f, 0, SEEK_END)){
		fprintf(stderr, "Unable to fseek %s\n", fname);
		return 0;
	}
	fflush(f);
	fsize = ftell(f);
	rewind(f);

	//buffer = (char *)malloc((fsize+1)*sizeof(char));
	checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
	fread(buffer, fsize, 1, f);
	fclose(f);
	*size = fsize;
	printf("fsize: %lu\n", fsize);
	return buffer;
}

int main(int argc, char **argv)
{
	int i;
	unsigned long temp;
	char * a_file;
	BYTE * buff;
	char option, index;

	// parse input
    while ((option = getopt(argc, argv,"f:")) != -1)
        switch (option) {
             case 'f' :
            	 a_file = optarg;
                 break;
             default:
            	 break;
        }

    // get number of arguments = files = jobs
    int n = argc - optind;

    JOB ** jobs;
    checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

    // iterate over file list
    for (i = 0, index = optind; index < argc; index++, i++){
        printf ("Non-option argument '%s'\n", argv[index]);
        buff = get_file_data(argv[index], &temp);
        jobs[i] = JOB_init(buff, temp);
    }

    print_jobs(jobs, n);
	pre_sha256();
	runJobs(jobs, n);
	cudaDeviceSynchronize();
	print_jobs(jobs, n);

	cudaDeviceReset();
    return 0;
}
