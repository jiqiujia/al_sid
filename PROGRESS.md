# 当前进展总结

> 项目：FORGE（Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets）
> 代码仓库：`/apdcephfs_sw/share_1546755/qiujiaji/gr/al_sid`（基于 `selous123/al_sid`）
> 总结时间：2026-10-14

## 一、项目目标

FORGE 是一个面向工业级数据集的**语义标识符（SID）构建与生成式检索（GR）基准**。核心工作分两段：

1. **SID 生成**：用 RQVAE（残差量化 VAE）把物品的多模态 512 维 embedding 量化成多层语义 ID（Codebook）。
2. **生成式检索**：用 Qwen2.5-0.5B / T5-base 等序列模型，基于用户历史行为序列预测下一个物品的 SID，并用 HitRate 评估。

数据集规模：140 亿用户交互、2.5 亿物品的多模态特征（淘宝电商场景）。线上实验取得 PVR +8.93%、成交笔数 +0.35% 的提升。

## 二、代码结构总览

```
al_sid/
├── SID_generation/          # 阶段一：SID（语义ID）生成
│   ├── configs/             # 配置（default.yml / rqvae_i2v.yml）
│   ├── rqvae_embed/         # RQVAE 核心模型（rqvae.py, rqvae_clip.py, quantizations.py, modules.py, layers.py）
│   ├── data_loader/         # 数据读取（local_reader_clip.py / local_reader_vqvae_reconstruct.py）
│   ├── utils/               # 配置、分布式、kmeans、logger、优化器等
│   ├── train.py             # 分布式训练入口
│   ├── infer_SID.py         # 推理生成 item→SID 映射
│   ├── scripts/             # gini_coefficient.md（SID 质量度量指标）
│   └── ID_collision.sql     # ID 冲突检测 SQL
├── algr/                    # 阶段二：生成式检索
│   ├── config/              # 训练/推理配置（t5base / qwen2.5，3layer，s1/tiny）
│   ├── models/              # t5 / qwen2_5 模型实现（modeling + data）
│   ├── utils/               # trainer、predict、data_collator、log 等
│   ├── runner.py            # 训练/预测主入口
│   ├── calc_hr.py           # HitRate 计算
│   └── item_info/           # item→SID 映射表（已下载）
├── data/                    # 已下载数据集
│   ├── AL-GR-v1/            # 行为序列（s1/s2/s3 splits + test_data）
│   └── Item-EMB/            # 物品多模态表征（base_feature / final_feature）
└── asset/                   # 图片资源（FORGE.png, Title.png）
```

## 三、已完成工作

### 1. 代码仓库就绪（Roadmap 全部勾选项）

Roadmap 中三项均已完成：

- [x] Generative Retrieval Training（生成式检索训练）
- [x] SID Generation（SID 生成）
- [x] Data Processing（数据处理）

代码库中已包含 RQVAE 模型、分布式训练脚本、推理脚本、GR 训练/推理 runner、HitRate 评估脚本，以及 Gini 系数等 SID 质量度量指标文档。

### 2. 数据集下载完成

已下载并落盘到工作目录（均为未纳入 git 的新增文件）：

- `data/AL-GR-v1/s1_splits/`：41 个分片（part_01 ~ part_41），每片约 6.14 GB，含用户行为序列。
- `data/AL-GR-v1/s2_splits/`：30 个分片，每片约 6.17 GB。
- `data/AL-GR-v1/s3_splits/`：31 个分片，每片约 6.15 GB。
- `data/AL-GR-v1/test_data/`：`s1_test.csv`(40 MB)、`s2_test.csv`、`s3_test.csv`，用于评估。
- `data/Item-EMB/`：物品 512 维多模态表征，含 `base_feature/`（原始版）与 `final_feature/`（预处理版，与 README 的 `final` 口径一致）。
- `SID_generation/datas/` 与 `datas_new/`：`contrastive_data_mask.npz`(13.54 GB)、`reconstruct_data_mask.npz`(7.99 GB)，用于 RQVAE 对比学习与重建任务。

### 3. Item→SID 映射表就位

`algr/item_info/` 下已放置两个版本的映射表（约 6 GB 级）：

- `tiny_item_sid_base.csv`（6.32 GB，基于 base codebook）
- `tiny_item_sid_final.csv`（6.03 GB，基于 final codebook，论文主结果所用）

### 4. 环境适配与配置修改（当前工作区未提交的改动）

针对本机环境做了以下适配（`git status` 显示为 modified / untracked）：

- **`SID_generation/infer_SID.py`**：将列名与路径占位符填为实际值——
  - `INPUT_EMBEDDING_COL = 'feature'`
  - `INPUT_ID_COL = 'base62_string'`
  - `INPUT_FILE_PATH = './item_feature/final_feature/part_1.csv'`
  - 即：准备用 `final` 版多模态表征（`part_1.csv`，2.54 GB）跑 SID 推理。
- **`algr/config/generate_qwen2.5_05b_3layer_tiny.json`**：`data_file` 改为本机 HF 缓存路径 `/root/.cache/huggingface/hub/datasets--AL-GR--AL-GR-Tiny/snapshots/.../test_data/s1_tiny_test.csv`。
- **`algr/config/qwen2.5_05b_3layer_s1tiny.json`**：`data_file` 改为本机 HF 缓存路径 `.../AL-GR--AL-GR-Tiny/snapshots/c160ac2d.../train_data/s1_tiny.csv`，`streaming: true`。

### 5. 数据文件就位（SID 推理输入）

- `SID_generation/item_feature/final_feature/part_1.csv`（2.54 GB）已放置，作为 `infer_SID.py` 的输入 embedding 文件。

## 四、当前进展定位

**当前处于「配置与数据准备完成 → 即将进入模型运行」的临界点**，具体判断依据：

- 全部数据集、映射表、embedding 文件均已下载落盘，无缺失。
- 三个配置/脚本文件已针对本机路径完成适配（`TODO` 占位符已消除）。
- **尚未生成任何训练/推理输出**：工作区内不存在 `logs/`、`output*`、checkpoint 等运行产物目录（`find` 仅命中 git 自身的 `.git/logs`），说明 `train.py`、`infer_SID.py`、`runner.py` 都还未实际执行（或执行结果未保留在本目录）。
- 未提交改动集中在"路径适配"，属于运行前的最后一步准备。

## 五、下一步计划

按如下顺序推进：

1. **SID 生成（可选，若复用官方映射表可跳过）**
   训练 RQVAE：
   ```bash
   python -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --master_port=27646 train.py --output_dir=/path/to/output --save_prefix=MODEL_NAME --cfg=configs/rqvae_i2v.yml
   ```
   再推理得到 item→SID 映射（当前 `infer_SID.py` 已配好输入路径，需指定 `CKPT_PATH`）：
   ```bash
   python infer_SID.py
   ```
   > 注意：`CKPT_PATH = 'output_model/checkpoint-7.pth'` 仍为示例值，需替换为实际训练出的 checkpoint。

2. **生成式检索训练（tiny 快速验证）**
   ```bash
   cd algr
   python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/qwen2.5_05b_3layer_s1tiny.json
   ```
   > 该 config 的 `load_checkpoint_from` 为 `Qwen/Qwen2.5-0.5B-Instruct`，需确保能访问 HF 或已本地缓存该模型。

3. **推理预测**
   ```bash
   python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/generate_qwen2.5_05b_3layer_tiny.json
   ```

4. **评估 HitRate**
   ```bash
   python calc_hr.py --item_sid_file=item_info/tiny_item_sid_final.csv --generate_file=logs/generate_qwen2.5_05b_3layer_tiny/output.jsonl --decoder_only
   ```

5. **全量训练（tiny 验证通过后）**
   ```bash
   nohup python -m torch.distributed.launch --nnodes=1 --nproc_per_node=16 runner.py --config=config/qwen2.5_05b_3layer_s1.json > logs/qwen2.5_05b_3layer_s1/log.txt 2>&1 &
   ```

## 六、待确认 / 风险点

- **checkpoint 缺失**：SID 推理依赖的 RQVAE checkpoint（`output_model/checkpoint-7.pth`）尚未就位。若不打算自行训练 RQVAE，可直接复用已下载的官方 `tiny_item_sid_final.csv` 映射表跳过该步。
- **GPU 资源**：`s1/s2/s3` 全量数据达数百 GB，tiny 验证建议优先；全量训练需 `--nproc_per_node=16` 的多卡环境。
- **模型可访问性**：Qwen2.5-0.5B-Instruct 需从 HF 拉取，确保网络或本地缓存可用。
- **未提交改动**：当前 3 个 modified 文件 + 3 个 untracked 目录（`SID_generation/item_feature/`、`algr/item_info/`、`data/`）尚未 commit，其中 `data/` 与 `item_info/` 体量极大，建议保持 gitignore 不入库。

## 七、参考资源

- 论文：FORGE（arXiv:2509.20904）
- 主数据集：[AL-GR / AL-GR-v1](https://huggingface.co/datasets/AL-GR/AL-GR-v1/)
- Tiny 数据集：[AL-GR / AL-GR-Tiny](https://huggingface.co/datasets/AL-GR/AL-GR-Tiny)
- Item-SID 映射：[AL-GR / Item-SID](https://huggingface.co/datasets/AL-GR/Item-SID)（`sid_final.csv` / `sid_base.csv`）
- 多模态表征：[AL-GR / Item-EMB](https://huggingface.co/datasets/AL-GR/Item-EMB)
- 代码仓库：https://github.com/selous123/al_sid

## 八、2026-10-18 复现推进记录（环境适配 + tiny 全链路跑通）

### 1. 环境关键结论

- 原 `al_sid/venv`（torch 2.6.0+cu124）与节点驱动 535.161.08 **不兼容**：单卡 CUDA 正常，但 8 卡 NCCL 报 `CUDA driver version is insufficient`（8 卡实测全挂）。
- 系统 python（torch 2.10.0+cu129）8 卡 NCCL 全通，但 transformers 5.3.0 太新（`LossKwargs` 已移除），且系统 `apex` 缺 `amp` 模块。
- **最终可用环境**：节点本地盘 `/root/work/venv`（`--system-site-packages` 继承系统 torch 2.10+cu129）+ transformers 4.51.2 + tokenizers 0.21.1 + stub `apex` 包 + `oss2==2.19.1`。
- `Qwen/Qwen2.5-0.5B-Instruct`（954MB）经 `star-proxy.oa.com:3128` 下载到 HF 缓存。容器无直连外网；`.bashrc` 里的 proxy 在非交互 shell 不生效，需显式 export。

### 2. 执行方式（重要，避免静默失败）

- `jizhi_client exec` 会话空闲会被回收；`nohup` 挡不住——torch elastic launcher 自行注册了 SIGHUP handler，覆盖 `nohup` 的 SIG_IGN。
- 短命令：`printf '0\n' | script -qec "taiji_client exec <TF> <IID> bash -c '<cmd>'" /dev/null` 一次性执行。
- 长任务：`setsid nohup <cmd> > /root/xxx.log 2>&1 < /dev/null &` 脱离控制终端。
- 启动后当场校验（日志存在 + `ps` 到进程 + 有预期首行输出）；监控要同时匹配成功标记和 `Traceback/Killed`。详见 iWiki 4048158351 新增「七」。

### 3. 代码/配置修正

- `algr/config/generate_qwen2.5_05b_3layer_tiny.json`：`load_checkpoint_from` 从 `./logs/qwen2_5_05b_3layer_tiny`（不存在）改为 `./logs/qwen2.5_05b_3layer_s1tiny/checkpoint-12500`。
- `calc_hr.py` 的 `pd.DataFrame(DatasetDict)` 会把 dict 键当列名、得到 1 列（`train`），导致 `df.columns` 赋值 4 列失败。绕法：`load_dataset("csv", ...)` 后用 `ds["train"].to_pandas()`。已用独立脚本预生成缓存 `algr/data/sid2item_v_tiny_item_sid_final.json`（255,289,109 行 → 248,715,255 个 SID，7.9GB），`calc_hr` 后续直接读缓存。

### 4. tiny 训练结果（qwen2.5_05b_3layer_s1tiny，8×H20）

- 12500 步 / 2h55m，1.19 it/s，761 samples/s；loss 6.08 → 4.07 单调下降。
- checkpoint：`algr/logs/qwen2.5_05b_3layer_s1tiny/checkpoint-12500`。

### 5. 推理 + HitRate（s1_tiny_test，9794 样本）

- 输出 `algr/logs/generate_qwen2.5_05b_3layer_tiny/output.jsonl`（276MB，9800 行 ≈ 9794 + DDP padding）。
- 每条样本生成 1200 个 SID：自定义 beam search 在 `num_return_sequences==num_beams` 时走 dynamic_beams，`beams_to_keep=(1+2个eos)×100=300`，`max_new_tokens=3` 三步解码末步保留 `4×300=1200` 个候选——这是算 HR@500/1000 所需，非 bug；候选按累积 log-prob 降序，口径正确。

| 指标 | 本次 tiny | 论文全量 Qwen2.5-0.5B-s1(base) |
|:--|:--|:--|
| HR@20 | 0.53% | 5.06% |
| HR@100 | 1.70% | 12.77% |
| HR@500 | 2.98% | 26.02% |
| HR@1000 | 3.12% | 30.68% |

> tiny 仅为管线冒烟验证（12500 步、欠拟合，末步 loss 4.07），最终指标需全量 s1 训练。

### 6. 下一步

- 全量 s1 训练：`algr/config/qwen2.5_05b_3layer_s1.json`（16 卡，数百 GB 数据）。
- GPU 占用 `multi_gpu_occupy.py` 已恢复（8 卡 100%），训练前需再次停掉。

## 九、2026-10-21~25 全量 s1 训练与端到端复现结果

### 1. 全量训练

- 配置：`algr/config/qwen2.5_05b_3layer_s1.json`（125000 步、lr 5e-5、batch 80、streaming）。
- 数据：`data/AL-GR-v1/s1_splits/part_01~41.csv`（246 GB）。
- 方式：**8 卡单节点**（16 卡双节点 NCCL 起不来，见下）。
- 结果：125000/125000 跑完，train_loss 1.93，末段 step loss 2.749（tiny 基线 4.075）。

### 2. 关键修复（重要，避免重踩）

1. **checkpoint 写 cephfs 崩溃**：`torch.save` 报 `unexpected pos 64 vs 0`（FUSE 写入不稳定），50000 步时崩过一次。→ `output_dir` 改本地盘 `/root/work/logs`，训完再拷回 cephfs。
2. **runner.py 不续训**：`self.trainer.train()` 不带参数，transformers 4.51 不会自动读 `resume_from_checkpoint`。→ 改为 `train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)`。
3. **续训数据回退卡死**：streaming 数据集续训会丢弃 `global_step×batch`（40000×640≈2560 万）样本，dataloader 卡住。→ 配置加 `ignore_data_skip: true`。

### 3. 双节点 16 卡结论（暂不可用）

- 跨节点 NCCL 初始化卡死（日志 `Detected kernel version 5.4.241 ... can cause the process to hang`）。用独立 NCCL all_reduce 测试复现：`init_process_group("nccl")` 不返回。
- 单节点 8 卡 NCCL 正常。故全量训练用 8 卡（1.19~1.73 step/s，累计约 32 小时）。

### 4. HitRate 结果（Qwen2.5-0.5B，3 层 SID，beam=100）

| 测试集 | HR@20 | HR@100 | HR@500 | HR@1000 |
|:--|:--|:--|:--|:--|
| tiny 冒烟（12500 步，s1_tiny_test） | 0.53% | 1.70% | 2.98% | 3.12% |
| 全量模型（s1_tiny_test，9794 条） | 2.87% | 7.66% | 15.90% | 19.67% |
| 全量模型（s1_test 全量测试集，25063 条） | 4.45% | **11.20%** | 23.01% | 27.44% |
| 论文官方 Forge-Qwen2.5-0.5B-Base-s1 | 5.06% | 12.77% | 26.02% | 30.68% |

- 全量模型在全量测试集上达到论文官方值的 **88~89%**，端到端复现成功。
- 差距来源：8 卡（论文 16 卡）导致等效 batch/更新节奏不同；官方卡列为 Base codebook 口径，而训练用 final codebook。

### 5. 产物

- 最终 checkpoint：cephfs `algr/logs/qwen2.5_05b_3layer_s1/checkpoint-125000`（6.3 GB）+ 本地盘 `/root/work/logs/qwen2.5_05b_3layer_s1/`。
- 推理输出：`algr/logs/generate_qwen2.5_05b_3layer_s1/output.jsonl`（720 MB，25063 样本）。
- sid2item 缓存：`algr/data/sid2item_v_tiny_item_sid_final.json`（2.487 亿 SID，7.9 GB）。
- 新增配置：`algr/config/generate_qwen2.5_05b_3layer_s1.json`（全量测试集 generate 配置）。

### 6. 环境（同「八」，最终可用）

- `/root/work/venv`：`--system-site-packages` 继承系统 torch 2.10.0+cu129 + transformers 4.51.2 + tokenizers 0.21.1 + stub apex + oss2。
- 节点驱动 535.161.08：与 venv 的 torch 2.6.0+cu124 不兼容（NCCL 报驱动不足），必须用系统 torch 2.10+cu129。
- 短命令 `script -qec` 一次性执行；长任务 `setsid nohup ... < /dev/null &`。

### 7. 双节点 16 卡训练配置（2026-10-26 已打通）

**结论**：双节点 16 卡可用，关键是 `NCCL_PXN_DISABLE=1`。

两节点各起一半（rank0=launcher `29.213.105.53`，rank1=worker-0 `29.206.0.253`），env 完全一致：

```bash
export NCCL_IB_GID_INDEX=3 NCCL_IB_SL=3 NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1 NCCL_SOCKET_IFNAME=bond1 UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0 SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2 NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1        # ← 关键，必须为 1

cd /apdcephfs_sw/share_1546755/qiujiaji/gr/al_sid/algr
/root/work/venv/bin/python -m torch.distributed.launch \
  --nnodes=2 --node_rank=<0|1> --nproc_per_node=8 \
  --master_addr=29.213.105.53 --master_port=29530 \
  runner.py --config=config/qwen2.5_05b_3layer_s1.json
```

**踩坑记录**：

1. `NCCL_PXN_DISABLE=0`（原配置）→ `ncclProxyClientGetFd ... failed : 3` + `ncclInternalError: operation=Connect, res=3`，在 DDP 参数同步（大张量）时崩溃。
2. **陷阱**：1 个元素的 `all_reduce` 测试能过（走轻量路径，不触发 PXN proxy），会误判为"通信正常"。必须用 **128MB 级大张量**测试才能复现。
3. `NCCL_PXN_DISABLE=1` 后，128MB all_reduce 16 rank 全过，真实训练稳定。

**验证脚本**：`/root/nccl3.py`（128MB all_reduce）+ `/root/nccl3_run.sh`（含上述 env）。

**性能**：16 卡约 2.4 it/s，8 卡约 1.19 it/s；B1 全量训练从 ~43h 缩短到 ~15h。

**注意**：worker-0 无 `python` 别名，用 `python3`；其本地盘也需要独立的 `/root/work/venv` 和 Qwen 模型缓存（不跨节点共享）。

### 8. 最终复现结果（16 卡，完整 1 epoch）

用 16 卡、`max_steps=125000`（等效 batch 1280）从头训练，跑满 **1 个完整 epoch**（`epoch: 1.0`，末步 loss 2.5408），与论文同口径。

| 模型 | 训练量 | HR@20 | HR@100 | HR@500 | HR@1000 |
|:--|:--|:--|:--|:--|:--|
| 8 卡（欠训） | 0.68 epoch | 4.45% | 11.20% | 23.01% | 27.44% |
| **16 卡（B1）** | **1.00 epoch** | **5.08%** | **12.70%** | **25.84%** | **30.59%** |
| 论文官方 Base-s1 | 1 epoch | 5.06% | 12.77% | 26.02% | 30.68% |

**复现比例 99.3%~100.4%，误差 <1%。** 结论：之前 88% 的差距完全来自欠训（0.68 vs 1.0 epoch），补齐训练量后即达到论文水平。

**产物**：最终 checkpoint 已覆盖 cephfs `algr/logs/qwen2.5_05b_3layer_s1/checkpoint-125000`；推理输出 `algr/logs/generate_qwen2.5_05b_3layer_s1/output.jsonl`（25063 样本）。
