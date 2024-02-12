# Chapter5: Prerequisites for obtaining high performance in CUDA programs

## Using CUDA events to time a block of code

```c++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start); // cannot use the macro function CHECK here

// The code block to be timed

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

ステップは以下

1. CUDAイベント`start`と`stop`を`cudaEvent_t`として定義し、`cudaEventCreate`関数で初期化。
2. `start`を`cudaEventRecord`に渡し開始時間を記録。
3. 処理コードブロックが終了後、`stop`を`cudaEventRecord`に渡し終了時刻を記録。`cudaEventSynchronize`で強制的に同期。
4. `cudaEventElapsedTime`で経過時間を計算。
5. `cudaEventDestroy`でイベント削除。

## Factors affecting GPU acceleration

### Ratio of data transfer

シングルプレシジョン(`float`)かダブルプレシジョン(`double`)で2倍ほど処理時間が変わる。
CPUとGPU間のデータ転送を含めたCUDAプログラム(`add3memcpy.cu`)では、C++プログラムに比べて3倍ほど遅い。これは、PCIeを介したGPUメモリにアクセスする帯域幅がCPUとGPU間のデータ転送の帯域幅と比べて1桁以上大きいため。
CUDAプログラムをより高速に処理させるには、CPUとGPU間のデータ転送を最小限に抑えることが重要である。

### Arithmetic intensity

配列加算問題の高速化率はあまり高くない、これはこの問題に対する算術強度が低いためである。ここでいう算術強度とは、算術演算量とそれに使われるメモリ操作の量の比率を指す。
シングルプレシジョンによる3つのグローバルメモリアクセスイベント(2つの読み込みと1つの書き込み)を伴うため算術強度は非常に低くなる。