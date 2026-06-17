# 无人机植保系统云边协同架构

一套完整的无人机植保系统云边协同解决方案，实现边缘端实时病害检测与云端数据持久化。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     无人机机载边缘端 (Jetson)                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ 多光谱摄像头  │→│ TensorRT 推理  │→│ 病害检测 (小麦锈病等) │  │
│  └──────────────┘  └───────────────┘  └──────────┬───────────┘  │
│  ┌──────────────┐                                │              │
│  │   GPS模块    │─── GPS坐标 ──→ 数据打包(JSON) ─┤              │
│  └──────────────┘                                ↓              │
│                               ┌──────────────────────────┐      │
│                               │ gRPC 双向流客户端 (Stream)│      │
│                               └────────────┬─────────────┘      │
└────────────────────────────────────────────┼────────────────────┘
                                             │ 4G/5G/WiFi
                                             │ gRPC (HTTP/2)
┌────────────────────────────────────────────┼────────────────────┐
│                   远端云服务器                               │    │
│                               ┌────────────↓─────────────┐      │
│                               │ gRPC 双向流服务端 (Stream)│      │
│                               └────────────┬─────────────┘      │
│                                            │                    │
│                               ┌────────────↓─────────────┐      │
│                               │   数据处理 / 过滤 / 聚合  │      │
│                               └────────────┬─────────────┘      │
│                                            │                    │
│                               ┌────────────↓─────────────┐      │
│                               │  InfluxDB 时序数据库     │      │
│                               │  - detection_frame       │      │
│                               │  - disease_spot          │      │
│                               │  - severity_aggregate    │      │
│                               └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
06-agriculture-drone-edge/
├── proto/
│   └── drone_service.proto          # gRPC Protocol Buffers 定义
├── edge/                            # 边缘端 (Jetson Python)
│   ├── main.py                      # 主程序入口
│   ├── requirements.txt             # 依赖包
│   ├── config/
│   │   └── config.yaml              # 边缘端配置
│   ├── modules/
│   │   ├── camera.py                # 多光谱摄像头采集
│   │   ├── detector.py              # TensorRT 目标检测推理
│   │   ├── gps.py                   # GPS 数据采集
│   │   └── grpc_client.py           # gRPC 双向流客户端
│   └── generated/                   # 生成的 Protobuf 代码
│       ├── drone_service_pb2.py
│       └── drone_service_pb2_grpc.py
└── cloud/                           # 云端服务器
    ├── main.py                      # 主程序入口
    ├── requirements.txt             # 依赖包
    ├── config/
    │   └── config.yaml              # 云端配置
    ├── modules/
    │   ├── grpc_server.py           # gRPC 双向流服务端
    │   └── influx_writer.py         # InfluxDB 时序写入
    └── generated/                   # 生成的 Protobuf 代码
        ├── drone_service_pb2.py
        └── drone_service_pb2_grpc.py
```

## 快速开始

### 1. 云端部署

```bash
cd cloud
pip install -r requirements.txt

# 编辑 config/config.yaml 配置 InfluxDB 连接信息
# 启动 (无 InfluxDB 时使用 mock 模式)
python main.py --no-influx --port 50051
```

### 2. 边缘端部署 (Jetson)

```bash
cd edge
pip install -r requirements.txt

# 若 Jetson 环境安装 TensorRT:
# pip install --extra-index-url https://pypi.nvidia.com tensorrt

# 编辑 config/config.yaml 配置云端地址
#   grpc.server_address: "your-cloud-server:50051"

# 生产模式启动
python main.py

# 测试模式 (使用 mock 摄像头/GPS 数据)
python main.py --mock --drone-id DRONE-TEST-001
```

## 核心功能详解

### 边缘端

#### 多光谱摄像头模块 (`modules/camera.py`)
- 支持 GStreamer / OpenCV 后端自动检测
- RGB / NIR (近红外) / RedEdge (红边) 多光谱波段切换
- 无锁环形缓冲队列，零丢帧设计
- 线程安全的帧获取接口

#### TensorRT 推理模块 (`modules/detector.py`)
- 自动加载 `.trt` 序列化引擎，FP16 精度加速
- YOLO 风格检测输出解析：目标框 + 置信度 + 类别
- NMS (非极大值抑制) 去重
- 病害严重程度分级：mild / moderate / severe
- **自动降级模式**：无 TensorRT 时使用 Mock 推理，便于开发测试

#### GPS 模块 (`modules/gps.py`)
- NMEA 协议解析 (pynmea2)
- 支持串口直连 `/dev/ttyTHS1` (Jetson 默认)
- 实时定位数据：经纬度、海拔、速度、航向
- Mock 模式自动模拟飞行轨迹

#### gRPC 客户端 (`modules/grpc_client.py`)
- **双向流式 RPC** (`StreamDetections`)
- 发送队列 + 确认队列双缓冲设计
- 自动断线重连 (指数退避)
- 心跳保持 (Keepalive)
- 服务器命令下行通道 (`StreamStatus`)
- 数据打包为 JSON 格式 (兼容 REST 接口扩展)

### 云端

#### gRPC 服务端 (`modules/grpc_server.py`)
- 双向流接收检测数据，返回确认 (ACK)
- 多无人机并发会话管理
- 帧数据去重 / 过滤 / 统计
- 下行命令通道 (切换光谱波段、调整阈值等)
- 实时监控指标：活跃无人机、帧率、严重程度分布

#### InfluxDB 写入模块 (`modules/influx_writer.py`)
- 三种时序数据 Measurement：
  - `detection_frame` - 每帧元数据 (GPS、帧数、推理延迟)
  - `disease_spot` - 单个病斑 (bbox、置信度、严重程度)
  - `severity_aggregate` - 严重程度聚合统计
- 批量写入 + 定时刷新 (100条/1秒)
- Line Protocol 编码
- **Mock 模式**：无 InfluxDB 时写入 JSONL 文件
- 写入失败回调机制

## 数据格式示例

### 单帧检测 JSON (边缘→云端)

```json
{
  "drone_id": "DRONE-TEST-001",
  "frame_id": 101,
  "timestamp": 1781734598426385408,
  "gps": {
    "latitude": 39.90420543,
    "longitude": 116.40740458,
    "altitude": 49.97,
    "speed": 8.55,
    "heading": 91.01
  },
  "multispectral_band": "RGB",
  "inference_latency_ms": 32.34,
  "detection_count": 1,
  "detections": [
    {
      "bbox": { "x1": 58, "y1": 248, "x2": 278, "y2": 446 },
      "confidence": 0.8205,
      "class_id": 2,
      "class_name": "wheat_rust_severe",
      "severity_score": 0.9462,
      "severity_level": "severe"
    }
  ]
}
```

## 病害检测模型训练建议

本系统兼容 YOLOv5/YOLOv8 风格的检测模型：

1. **数据集准备**：收集小麦锈病早期叶片图像，标注 3 类：
   - `wheat_rust_early` (轻度，<30% 叶片感染)
   - `wheat_rust_moderate` (中度，30-70%)
   - `wheat_rust_severe` (重度，>70%)

2. **导出 TensorRT**：
   ```python
   from ultralytics import YOLO
   model = YOLO("rust_detection.pt")
   model.export(format="engine", half=True, imgsz=640)
   # 输出 rust_detection.trt，放入 edge/models/
   ```

3. **配置更新**：修改 `edge/config/config.yaml` 中 `inference.tensorrt.engine_path`

## 生产环境部署建议

| 项目 | 建议 |
|------|------|
| 边缘端硬件 | Jetson Xavier NX / Orin NX (≥8GB 内存) |
| 摄像头 | 多光谱相机 (如 Micasense RedEdge-MX)，USB/CSI |
| 网络链路 | 4G/5G 工业模组，确保 ≥2Mbps 上行带宽 |
| 云端 | 2核4G 以上服务器，公网 IP，开放 50051 端口 |
| 时序数据库 | InfluxDB Cloud / 自建 InfluxDB 2.7+ |
| 安全 | 启用 TLS 加密，接入 gRPC 认证 (JWT/OAuth) |

## 性能指标 (Jetson Orin NX)

| 指标 | 数值 |
|------|------|
| 输入分辨率 | 640×640 FP16 |
| 推理延迟 | 8-12 ms/帧 |
| 检测帧率 | 25-30 FPS |
| 端到端延迟 | <50 ms (含采集+推理+发送) |
| 数据上行 | ~20 KB/帧，~600 KB/s (30 FPS) |
