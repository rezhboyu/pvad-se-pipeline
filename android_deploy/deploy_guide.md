# SE(GTCRN) + pVAD Android 部署指南

## 架構概覽

```
AudioRecord (16kHz, mono, 256 samples/frame = 16ms)
    │
    ▼
┌─────────────────────────────┐
│  GTCRN 串流降噪              │  ← gtcrn_simple.onnx (523KB)
│  nfft=512, hop=256          │
│  輸入: STFT frame (257, 2)  │
│  輸出: Enhanced STFT (257,2)│
│  延遲: 16ms (1 frame)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  CachedPVAD                  │  ← wespeaker_resnet34.onnx (26MB)
│  每 32 幀 (0.5s) 提取一次    │     enrollment 時 + 每 0.5s 各一次
│  d-vector embedding          │
│  cosine similarity > 閾值    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Soft Gate                   │  ← 純算術，無模型
│  attack=5ms, release=50ms   │
│  gain_floor=0.05 (-26dB)    │
└─────────────┬───────────────┘
              │
              ▼
AudioTrack (16kHz, mono, 256 samples/frame)
```

## 1. 依賴套件

### ONNX Runtime Android

```groovy
// build.gradle (app)
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
}
```

版本建議：1.17.0+，支援 NNAPI delegate（可選）和 XNNPACK（CPU 加速）。

### 最低系統需求

- Android API 26+ (Android 8.0)
- arm64-v8a（推薦）或 armeabi-v7a
- 記憶體：~100MB 可用 RAM

## 2. 模型檔案

放置位置：`app/src/main/assets/models/`

| 檔案 | 大小 | 用途 | 載入時機 |
|------|------|------|----------|
| `gtcrn_simple.onnx` | 523 KB | 串流降噪 | App 啟動時 |
| `wespeaker_resnet34.onnx` | 26 MB | Speaker embedding | App 啟動時 |

總模型大小：~27 MB（APK 內或首次啟動下載）

## 3. 音頻 I/O 設定

### AudioRecord（麥克風輸入）

```java
int sampleRate = 16000;
int channelConfig = AudioFormat.CHANNEL_IN_MONO;
int audioFormat = AudioFormat.ENCODING_PCM_FLOAT;  // 或 PCM_16BIT
int bufferSize = 256;  // GTCRN hop size = 256 samples = 16ms

AudioRecord recorder = new AudioRecord(
    MediaRecorder.AudioSource.MIC,
    sampleRate,
    channelConfig,
    audioFormat,
    bufferSize * 4  // 至少 4 倍 buffer 避免 underrun
);
```

### AudioTrack（輸出播放）

```java
AudioTrack track = new AudioTrack.Builder()
    .setAudioAttributes(new AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build())
    .setAudioFormat(new AudioFormat.Builder()
        .setSampleRate(16000)
        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build())
    .setBufferSizeInBytes(256 * 4 * 4)
    .setTransferMode(AudioTrack.MODE_STREAM)
    .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
    .build();
```

### 低延遲提示

- 使用 `PERFORMANCE_MODE_LOW_LATENCY`
- 考慮 AAudio / Oboe 取代 AudioRecord/AudioTrack
- 處理線程設為 `THREAD_PRIORITY_URGENT_AUDIO`

## 4. 模型載入與推論流程

### 4.1 初始化（Application/Service onCreate）

```java
// 1. 建立 ONNX Runtime 環境
OrtEnvironment env = OrtEnvironment.getEnvironment();

// 2. 載入 GTCRN（輕量，523KB）
OrtSession.SessionOptions gtcrnOpts = new OrtSession.SessionOptions();
gtcrnOpts.setIntraOpNumThreads(1);
// 可選: gtcrnOpts.addNnapi();  // 啟用 NNAPI
OrtSession gtcrnSession = env.createSession(
    loadModelFromAssets("models/gtcrn_simple.onnx"), gtcrnOpts
);

// 3. 載入 WeSpeaker（較大，26MB，建議異步載入）
OrtSession.SessionOptions speakerOpts = new OrtSession.SessionOptions();
speakerOpts.setIntraOpNumThreads(1);
OrtSession speakerSession = env.createSession(
    loadModelFromAssets("models/wespeaker_resnet34.onnx"), speakerOpts
);

// 4. 初始化 GTCRN caches
float[] convCache = new float[2 * 1 * 16 * 16 * 33];   // 零初始化
float[] traCache = new float[2 * 3 * 1 * 1 * 16];
float[] interCache = new float[2 * 1 * 33 * 16];
```

### 4.2 Enrollment（一次性）

```java
// 從錄製的 enrollment 音頻提取 d-vector
float[] enrollmentAudio = recordEnrollment();  // 3-5 秒
float[] fbank = computeKaldiFbank(enrollmentAudio);  // (T, 80)
OnnxTensor fbankTensor = OnnxTensor.createTensor(env,
    FloatBuffer.wrap(fbank), new long[]{1, T, 80});
float[] enrollmentDvector = speakerSession.run(
    Collections.singletonMap("feats", fbankTensor)
).get(0).getValue();
// L2 normalize
normalizeL2(enrollmentDvector);
```

### 4.3 串流處理迴圈

```java
// AudioRecord callback (每 16ms)
int frameCount = 0;
float cachedSim = 0f;
boolean cachedIsTarget = false;
int PVAD_INTERVAL = 32;  // 每 0.5s

void onAudioFrame(float[] newSamples) {  // 256 samples
    // 1. GTCRN 降噪
    float[] mixInput = prepareGTCRNInput(newSamples);  // STFT → (1,257,1,2)
    Map<String, OnnxTensor> gtcrnInputs = Map.of(
        "mix", createTensor(mixInput, new long[]{1, 257, 1, 2}),
        "conv_cache", createTensor(convCache, ...),
        "tra_cache", createTensor(traCache, ...),
        "inter_cache", createTensor(interCache, ...)
    );
    OrtSession.Result result = gtcrnSession.run(gtcrnInputs);
    float[] enhSpec = result.get("enh").getValue();
    convCache = result.get("conv_cache_out").getValue();
    traCache = result.get("tra_cache_out").getValue();
    interCache = result.get("inter_cache_out").getValue();
    float[] denoised = istft(enhSpec);  // → 256 samples

    // 2. CachedPVAD（每 32 幀才跑一次 WeSpeaker）
    frameCount++;
    if (frameCount % PVAD_INTERVAL == 0) {
        float[] windowAudio = getLastNSamples(8000);  // 0.5s buffer
        float[] fbank = computeKaldiFbank(windowAudio);
        float[] embedding = runSpeakerEncoder(fbank);
        normalizeL2(embedding);
        cachedSim = cosineSimilarity(embedding, enrollmentDvector);
        cachedIsTarget = cachedSim > THRESHOLD;
    }

    // 3. Soft gate
    float[] output = applySoftGate(denoised, cachedIsTarget, cachedSim);

    // 4. 寫入 AudioTrack
    audioTrack.write(output, 0, 256, AudioTrack.WRITE_NON_BLOCKING);
}
```

## 5. 記憶體與算力估算

### 模型記憶體

| 元件 | 模型大小 | 推論記憶體 | 備註 |
|------|---------|-----------|------|
| GTCRN | 523 KB | ~2 MB | 包含 caches |
| WeSpeaker ResNet34 | 26 MB | ~50 MB | 含中間 activations |
| 音頻緩衝區 | - | ~0.5 MB | input/output buffers |
| **總計** | **~27 MB** | **~53 MB** | |

### 每幀算力（Python CPU 測量值）

| 元件 | 平均延遲 | P95 延遲 | 頻率 |
|------|---------|---------|------|
| GTCRN 降噪 | ~0.6 ms | ~1.0 ms | 每幀 (62.5 fps) |
| WeSpeaker 提取 | ~30 ms | ~40 ms | 每 32 幀 (2 fps) |
| Soft gate | <0.01 ms | <0.01 ms | 每幀 |
| **每幀平均** | **~1.5 ms** | **~3 ms** | **在 16ms 預算內** |

### Android CPU 估算

Python ONNX Runtime 的速度通常比 Android CPU 快 2-3x（因為桌面 CPU 更強）。

保守估算 Android 端（Snapdragon 7xx 系列）：

| 元件 | 估算延遲 |
|------|---------|
| GTCRN 降噪 | ~1.5 ms |
| WeSpeaker (每 32 幀) | ~80 ms → 均攤 ~2.5 ms/frame |
| **每幀總計** | **~4 ms** |
| **RTF** | **~0.25** |

結論：即使在中階手機上，16ms 的即時預算也綽綽有餘。

### NNAPI / GPU Delegate

- GTCRN 的算子應大部分相容 NNAPI，可進一步加速
- WeSpeaker ResNet34 的 Conv2d 也能受益於 NNAPI
- 啟用方式：`sessionOptions.addNnapi()` 或 `sessionOptions.addXnnpack()`

## 6. 延遲分析

```
端到端延遲 = AudioRecord 延遲 + 演算法延遲 + AudioTrack 延遲

AudioRecord:  ~10-20 ms (取決於 buffer 設定)
演算法延遲:
  - GTCRN:    16 ms (1 frame hop) + ~1.5 ms (推論)
  - pVAD:     0 ms (非同步，不在關鍵路徑上)
  - Gate:     <0.1 ms
AudioTrack:   ~10-20 ms

總延遲:  ~38-58 ms
```

對於語音通訊應用，<100ms 的延遲被認為是「優秀」的。

## 7. 檔案清單

```
android_deploy/
├── models/
│   ├── gtcrn_simple.onnx          # 523 KB - 串流降噪
│   └── wespeaker_resnet34.onnx    # 26 MB  - speaker embedding
├── simulate_android_streaming.py   # Python 模擬 demo
├── deploy_guide.md                 # 本文件
└── performance_report.json         # 效能測試報告（執行 demo 後產生）
```

## 8. 注意事項

1. **WeSpeaker 模型不能裁剪**：ResNet34 的結構不適合 pruning，但可以考慮量化（INT8）來減小 APK 大小
2. **GTCRN 已經很小**：523KB 的模型幾乎不需要優化
3. **pVAD 間隔可調**：`PVAD_INTERVAL=32` (0.5s) 是平衡精度和效能的預設值，可根據需求調整
4. **Enrollment 音頻品質**：建議在安靜環境下錄製 3-5 秒的 enrollment，品質直接影響 pVAD 準確度
5. **電池消耗**：連續運行時，主要消耗來自 AudioRecord + GTCRN 推論，估計額外消耗 ~100mW
