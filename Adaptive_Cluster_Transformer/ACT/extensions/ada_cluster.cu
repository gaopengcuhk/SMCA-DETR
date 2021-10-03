#include <torch/extension.h>

typedef torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> int64_2d;
typedef torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> int_2d;

__global__ void ada_cluster_kernel(
    const int64_2d hashes,
    const int64_2d pos,
    int_2d groups,
    int_2d counts, 
    int L, 
    int N
){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int c = 0;
    if (n >= N) return;
    groups[n][pos[n][0]] = c;
    counts[n][c]++;
    for (int i = 1; i < L; i++) {
        if (hashes[n][pos[n][i]] != hashes[n][pos[n][i-1]]) {
            c++;
        }
        groups[n][pos[n][i]] = c;
        counts[n][c]++;
    }
}

void ada_cluster(
    const torch::Tensor hashes,
    const torch::Tensor pos,
    torch::Tensor groups,
    torch::Tensor counts
){
    int N = hashes.size(0);
    int L = hashes.size(1);
    
    const int threads = 1024;
    int blocks = (N - 1) / threads + 1;

    ada_cluster_kernel<<<blocks, threads>>>(
        hashes.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        pos.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        groups.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(), 
        L, N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_ada_cluster", &ada_cluster);
}
