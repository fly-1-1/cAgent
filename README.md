# cAgent — On-Device LLM Chat (Android)

本地离线大模型聊天 App，支持高通 Adreno GPU 加速推理。

## 功能

- 完全离线运行，无需网络
- 从设备本地导入 GGUF 格式模型
- GPU 加速推理（OpenCL / Qualcomm Adreno）
- 流式输出，逐 token 显示
- 支持中英文对话
- ChatML 格式 prompt（兼容 Qwen 系列模型）

## 技术栈

| 层级 | 技术 |
|------|------|
| UI | Jetpack Compose + Material 3 |
| 推理引擎 | llama.cpp (C++) |
| GPU 后端 | OpenCL (Adreno 优化 kernel) |
| JNI 桥接 | Kotlin ↔ C++ via NDK |
| 构建 | Gradle + CMake |

## 系统要求

- Android 14+ (API 34)
- 12–16 GB RAM
- 推荐：Qualcomm Snapdragon 8 Gen3 / 8 Elite（Adreno 750/830）

## 模型推荐

| 模型 | 大小 | 速度 (GPU) |
|------|------|-----------|
| Qwen2.5-1.5B Q4_0 | ~900 MB | 快 |
| Qwen2.5-3B Q4_0 | ~1.7 GB | 中 |
| Qwen2.5-7B Q4_K_M | ~4.5 GB | 较慢 |

纯 Q4_0 量化在 Adreno GPU 上性能最佳（有专用优化 kernel）。

## 构建

### 前置条件

- Android Studio (最新版)
- Android NDK 28+
- CMake 3.22+
- Python 3（用于 OpenCL kernel 嵌入）

### 步骤

```bash
# 克隆（含子模块）
git clone --recursive <repo-url>
cd cAgent

# 用 Android Studio 打开项目，或命令行构建
./gradlew assembleDebug

# 安装到设备
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 项目结构

```
app/src/main/
├── cpp/
│   ├── CMakeLists.txt        # Native 构建配置
│   ├── native-lib.cpp        # JNI 桥接 (llama.cpp 调用)
│   └── opencl_stub.c         # OpenCL 运行时加载器
├── java/com/example/cagent/
│   ├── llm/LlmEngine.kt     # Native 方法声明
│   ├── model/ChatMessage.kt  # 消息数据类
│   ├── storage/              # 模型存储管理
│   └── ui/                   # Compose UI (Chat/Models)
third_party/
├── llama.cpp                 # LLM 推理引擎 (submodule)
└── OpenCL-Headers            # Khronos OpenCL 头文件 (submodule)
```

## GPU 加速原理

App 使用 llama.cpp 的 OpenCL 后端，该后端由高通团队参与开发并在 Snapdragon 8 Gen3/Elite 上验证。

由于 Android 12+ 的 linker namespace 限制，第三方 app 无法直接链接 vendor 的 `libOpenCL.so`。本项目通过以下方式解决：

1. `AndroidManifest.xml` 声明 `<uses-native-library android:name="libOpenCL.so">`
2. 自定义 `opencl_stub.c` 作为运行时加载器，在库加载时（主线程）通过 `dlopen` 绝对路径加载 vendor 驱动
3. 所有 OpenCL 函数指针在 `__attribute__((constructor))` 中一次性 resolve，避免 engine 线程栈溢出

如果设备不支持 OpenCL，自动回退到 CPU 推理。

## 使用方法

1. 安装 App
2. 下载 GGUF 模型到手机（推荐 Qwen2.5-1.5B Q4_0）
3. 打开 App → Models 页面 → "Import GGUF from device"
4. 切换到 Chat 页面 → 开始对话

## License

MIT
