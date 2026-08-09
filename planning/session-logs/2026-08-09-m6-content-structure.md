# 2026-08-09 M6 九章内容与结构补强

## 目标

相对 OpenMLSys 补强硬件/GPU 叙事、编译栈、部署闭环、接口史/计算图、
章末收束与可选真机跑通文档；默认可运行路径保持 CPU（D022）。

## 操作

### M6a

1. AUTHORING：GPU 三轨 +「本章系统结论」体例。
2. 九章练习页插入系统结论；ch3/4/6/7/9 着陆页进阶台阶句。
3. 设备/Runtime 地图（`ch01/04-burn-stack.md` +
   `img/ch01-device-runtime-map.svg`）；控制面/数据面图。

### M6b

1. ch3：多 Runtime 表、GPU 心智模型锚定、产业短表；CPU/WGPU 实验边界。
2. ch4：前后端分层、Pass→Fusion→多 Runtime 图、GPU sync 检查清单。
3. ch2：接口简史、Module→Kernel 层次、tape vs Fusion 图、产业短表。
4. ch7：部署闭环图 + Device 与 artifact 正交；推理产业短表。
5. ch6：collective 源码导读顺序 + 产业短表；ch9：AllReduce 机柜数值例。

### M6c

1. 新建 `docs/OPTIONAL_PROFILES.md`（wgpu / onnx-fixture / 未来 cuda）。
2. 新增 D022；更新 `running-examples.md`、AUTHORING、STATUS。

## 验证

见 STATUS「本次交接」。

## 下一步

提交推送（若发布者要求）；候选 tag 仍不阻塞于 M6；真机 CUDA/NCCL
仅在 pins 与环境允许时追加 profile 命令。
