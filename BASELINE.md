# 纯分割 CPU 开发基线

环境：Windows x64、Python 3.12.8、PyTorch 2.13.0+cpu。
核心依赖独立于原有完整研究依赖表，避免默认安装 CUDA 11.0 CuPy 等可选组件。

## 安装和验证

在项目根目录执行 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_smoke.py -q
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache\matplotlib'
.\.venv\Scripts\python.exe scripts/train.py --help
```

核心包使用固定版本；完整已安装环境见 `requirements-cpu-lock.txt`，其中包含传递依赖。
重建相同环境时，先按上面命令安装 CPU torch，再安装完整锁定文件。
GPU 环境需要另行匹配硬件、驱动及 PyTorch wheel，本基线仅验证 CPU。

2026-09-10 本机验证：`pip check` 通过；三项 pytest 测试全部通过（18.83 秒）；
训练入口 `--help` 和 YAML 配置读取通过；锁定文件的 42 个包与安装环境逐项一致。

## 真实 BraTS 运行

将已获授权的数据放到配置的 `data.data_dir`，每个病例目录必须包含四个模态和一个标签：

```text
data/BraTS2021/BraTS2021_00000/
├── BraTS2021_00000_t1.nii.gz
├── BraTS2021_00000_t2.nii.gz
├── BraTS2021_00000_flair.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
└── BraTS2021_00000_seg.nii.gz
```

训练入口现在会先按病例ID做互斥的 train/validation/test 拆分；训练增强只挂在训练
Dataset 上。标签值按 BraTS 约定将 `4` 映射为四分类索引 `3`，缺少标签或出现其他标签值
会直接报错。每个 epoch 结束后使用 validation loss 选择 `best_model.pt`，全部 epoch 完成
后才在独立 test split 上运行并把结果写入 `training_history.json`。

真实训练命令：

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache\matplotlib'
.\.venv\Scripts\python.exe scripts/train.py --config config/experiment.yaml
```

仓库当前没有 `data/BraTS2021`，所以本机尚未产生真实病例的 loss、Dice、IoU 或 checkpoint；
`tests/test_brats_data.py` 使用临时 NIfTI 病例验证了同一数据接口和一轮 train/validation/test
控制流，但这些数值不代表 BraTS 实验结果。
锁定文件记录实际安装版本，未在第二个全新环境中再次安装验证。

## 修复和测试范围

首个故障在 `dec4` 的 skip 注意力卷积：小模型期望 16 通道，编码器实际输出 32 通道。
四级解码器现与实际 skip 通道对齐；各辅助头连接对应解码特征，并插值到主输出空间尺寸。
`return_features` 返回实际瓶颈和各级解码特征。
解码器与输出头参数形状发生变化，旧 checkpoint 不能直接按原结构加载。

测试输入为随机 `1×4×32×32×32`，标签索引为 0–3，`base_filters=2`。
测试覆盖输出形状、辅助头形状、有限损失/梯度、优化器参数更新，以及项目
`CombinedLoss` 与 `SegmentationTrainer.train_epoch` 的单批次训练。
测试日志放入 pytest 临时目录。

这只证明合成张量训练步骤可运行，不证明真实 BraTS 数据训练、完整验证循环或模型性能。
仍待处理：BraTS 4→3 标签映射、MONAI 数据变换兼容性与空间处理、共享数据集增强覆盖、
验证指标调用、最佳 checkpoint 选择，以及真正接入多模态融合。
当前 `attention` 配置中的模态融合模块仍未参与 forward；测试不作为融合策略验证。
