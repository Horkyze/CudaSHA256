#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>


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


void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB * JOB_init(BYTE * data, long size, char * fname) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}


BYTE * get_file_data(char * fname, unsigned long * size) {
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
	return buffer;
}

void print_usage(){
	printf("/.CudaSHA256 <file> ...\n");
}

int main(int argc, char **argv) {
	int i = 0, n = 0;
	size_t len;
	unsigned long temp;
	char * a_file = 0, line = 0;
	BYTE * buff;
	char option, index;
	JOB ** jobs;

	// parse input
	while ((option = getopt(argc, argv,"hf:")) != -1)
		switch (option) {
			case 'h' :
				print_usage();
				break;
			case 'f' :
				a_file = optarg;
				break;
			default:
				break;
		}


	if (a_file) {

		checkCudaErrors(cudaMallocManaged(&jobs, 1001 * sizeof(JOB *)));

		DIR * d;
		struct dirent * dir;
		d = opendir(a_file);
		if (d) {
			while ((dir = readdir(d)) != NULL){
				//printf("%s\n", dir->d_name);
				if (dir->d_name[0] != '.') {
					buff = get_file_data(dir->d_name, &temp);
					jobs[n++] = JOB_init(buff, temp, dir->d_name);
				}

			}
		  closedir(d);
		}


		pre_sha256();
		runJobs(jobs, n);

		// FILE * f = 0;
		// f = fopen(fname, "rb");
		// if (!f){
		// 	fprintf(stderr, "Unable to open %s\n", fname);
		// 	return 0;
		// }
		// while ((read = getline(&line, &len, f)) != -1) {
		// 	printf("Retrieved line of length %zu :\n", read);
		// 	printf("%s", line);
		// }

	} else {
		// get number of arguments = files = jobs
		n = argc - optind;
		if (n > 0){

			checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

			// iterate over file list - non optional arguments
			for (i = 0, index = optind; index < argc; index++, i++){
				buff = get_file_data(argv[index], &temp);
				jobs[i] = JOB_init(buff, temp, argv[index]);
			}

			//print_jobs(jobs, n);
			pre_sha256();
			runJobs(jobs, n);
		}
	}

	cudaDeviceSynchronize();
	print_jobs(jobs, n);
	cudaDeviceReset();
	return 0;
}
