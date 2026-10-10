这是一份按章维护的 CUDA 速查笔记，用于学完后快速回忆。每章只保留核心模型、必须记住的代码、易混点和自测题；具体推导与完整示例回到原书查看。

## 第 1 章 GPU 硬件与 CUDA 程序开发工具

### 核心模型

- CUDA 是 CPU 与 GPU 协作的异构计算。

- CPU 是 host：负责控制、调度和复杂逻辑。

- GPU 是 device：负责大规模、结构相似的并行计算。

- 独立 GPU 通常有自己的显存；主机内存与显存之间的数据传输可能成为性能瓶颈。

### 必记概念

- CPU：少量强核心，擅长低延迟、复杂控制和串行任务。

- GPU：大量计算核心，擅长高吞吐量并行任务。

- Compute Capability X.Y：表示 GPU 的硬件功能级别，不等于性能。

- 衡量 GPU 不能只看 TFLOPS，还要看计算精度、显存带宽、显存容量和 ECC。

- CUDA Runtime API 更易使用；Driver API 更底层、控制更灵活。

- GPU、NVIDIA Driver、CUDA Toolkit 和主机编译器必须相互兼容。

### 环境速查

```powershell
nvidia-smi       # GPU、驱动、显存、利用率等
nvcc --version   # 本机 CUDA Toolkit 编译器版本
```

`nvidia-smi` 中的 CUDA Version 表示驱动支持的 CUDA 用户态版本上限，不一定是本机 Toolkit 版本。多卡环境可用 `CUDA_VISIBLE_DEVICES` 控制程序可见的 GPU。

### 一句话复习

GPU 的优势来自大量线程带来的总体吞吐量；是否真正加速，还取决于并行度、数据搬运和内存访问。

### 自测

- Host 和 device 分别负责什么？

- Compute Capability 为什么不能直接代表性能？

- `nvidia-smi` 与 `nvcc --version` 分别检查什么？

---

## 第 2 章 CUDA 中的线程组织

### 最小 CUDA 程序

```cpp
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

- `__global__` 声明 kernel：通常由 host 调用、在 device 执行。

- kernel 返回类型必须是 `void`。

- `<<<grid_size, block_size>>>` 是执行配置。

- kernel 启动对 host 通常是异步的；`cudaDeviceSynchronize()` 等待设备完成，并使运行错误在同步点暴露。

### 线程层级

```text
Grid
└─ Thread Block
   └─ Thread
```

一维配置中：

```cpp
kernel<<<grid_size, block_size>>>();
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

- `gridDim`：网格包含多少个 block。

- `blockDim`：每个 block 包含多少个 thread。

- `blockIdx`：当前 block 的索引。

- `threadIdx`：当前 thread 在 block 内的索引。

- 一维总线程数：`gridDim.x * blockDim.x`。

### 多维配置

```cpp
dim3 grid(Gx, Gy, Gz);
dim3 block(Bx, By, Bz);
kernel<<<grid, block>>>();
```

线程块内的线性索引：

```cpp
int tid = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x
        + threadIdx.x;
```

`x` 维变化最快。二维数据常直接计算坐标：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

### Warp 与调度

- 一个 block 会被划分为 warp。

- 当前 CUDA GPU 的一个 warp 通常包含 32 个连续线程。

- GPU 以 warp 为主要调度单位，线程数不是 32 的倍数时，最后一个 warp 会有空闲 lane。

- 不同 block 的执行先后次序不能依赖；程序必须保证 block 之间相互独立。

### 常见限制

- 一个 block 的线程总数通常不能超过 1024。

- 还必须同时满足 x、y、z 各维度限制。

- 具体上限由设备决定，不要只依赖书中的旧架构表格。

- block 越大不一定越快；资源占用和 warp 数量都会影响实际性能。

### nvcc 编译模型

`nvcc` 将主机代码交给主机 C++ 编译器，并把设备代码编译为 PTX 和 GPU 二进制代码。

- `compute_XY`：虚拟架构和功能集合，对应 PTX。

- `sm_XY`：真实 GPU 架构，对应可直接加载的 cubin/SASS。

- 可嵌入多个 `sm_*` 版本形成 fat binary。

- 保留 PTX 后，新 GPU 可在运行时 JIT 编译。

- 目标架构越多，兼容范围越广，但编译时间和二进制体积也越大。

常用命令：

```bash
nvcc hello.cu -o hello
nvcc hello.cu -arch=sm_XY -o hello
nvcc --list-gpu-code
nvcc --list-gpu-arch
```

### 易混点

- `grid_size` 是 block 数，不是线程总数。

- `block_size` 是每个 block 的线程数。

- `threadIdx.x` 只在 block 内唯一；全局索引必须结合 `blockIdx.x`。

- 大量线程并不等于同时执行；硬件会分批调度。

- `warp` 属于一个 block，不会跨 block 组成。

- `nvcc` 能编译纯主机 C++，但只有调用 kernel 才真正使用 GPU。

### 一句话复习

Kernel 启动时创建一个 grid；grid 由 blocks 构成，block 由 threads 构成。每个线程通过内建索引找到自己负责的数据，硬件再把相邻线程按 warp 调度。

### 自测

- `kernel<<<80, 256>>>()` 一共创建多少线程？

- 为什么只用 `threadIdx.x` 不能获得整个 grid 中的唯一线程编号？

- `compute_XY` 和 `sm_XY` 分别代表什么？

- 为什么不同 block 的输出顺序可能变化？