#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void countKeys(int *key, int n, int range, int *bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
      atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void reconstructArray(int *key, int n, int range, int *bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
      int start_index = 0;
      for (int j = 0; j < i; ++j) {
          start_index += bucket[j];
      }
      int count = bucket[i];
      for (int k = 0; k < count; ++k)
      {
           int index = start_index + k;
           if(index < n){
              key[index] = i;
           }
      }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  int *bucket_device;
  int *key_device;
  cudaMallocManaged(&key_device, n * sizeof(int));
  cudaMallocManaged(&bucket_device, range * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  for (int j=0; j<range; j++) {
    bucket_device[j] = 0;
  }
  printf("\n");

  cudaMemcpy(key_device, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  countKeys<<<1, 64>>>(key_device, n, range, bucket_device);
  cudaDeviceSynchronize();

  reconstructArray<<<1, 64>>>(key_device, n, range, bucket_device);
  cudaDeviceSynchronize();

  cudaMemcpy(key.data(), key_device, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key_device);
  cudaFree(bucket_device);
}
