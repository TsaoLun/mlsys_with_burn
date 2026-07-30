# 第 7 章 模型服务

## 本章问题

训练产物如何转换、验证、优化并部署到服务器、浏览器或受限设备？

## 计划内容

- 训练与推理的系统差异
- Record、Burnpack 与模型状态
- burn-onnx 的 IR 和 Rust 代码生成
- 批处理、延迟、吞吐与服务接口
- Burn Remote 和异构客户端
- WASM、no_std 与边缘部署

## 实验

导入一个小型 ONNX 模型，验证输出，再将推理包装为最小服务。

## 来源与改编说明

计划参考 OpenMLSys v1 `chapter_model_deployment/`。转换和部署路径将
按 burn-onnx、Burn Record/Remote 与 Rust 目标平台重写。

