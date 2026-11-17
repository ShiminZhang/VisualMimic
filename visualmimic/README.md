# VisualMimic 第一阶段架构

接口以 SuperMimic 执行为核心，设计见 [INTERFACES.md](INTERFACES.md)。真实 TMED 编辑、GMR 重定向、SuperMimic 参考编译／仿真和轨迹渲染适配器已加入。完整第一阶段仍未验收：视频入口缺 SMPL-X 资产，官方人体编辑评测与两路对照尚未打通。

## 机器人执行模块与当前证据

- `backends/motionfix_editor.py`：加载官方 TMED 权重，已真实生成样本 000004 的 120 帧编辑结果。采用源动作长度，不能等同于使用目标长度的官方评测协议。
- `human_motion.py`、`retargeting.py`：人体表示到 G1；整段使用固定地面平移，保留跳跃／蹲起高度变化。地面估计假设片段包含脚接地，不能用于完全腾空片段。
- `reference_compiler.py`：通过 SuperMimic 原生重采样和 MuJoCo 正运动学生成参考。
- `supermimic_executor.py`、`execute_reference.py`：冻结策略真实仿真，关闭自动重置，保存终止状态及 K 个动作对应的 K+1 个状态。
- `tracking_evaluation.py`：输出覆盖率、全局位置／姿态／关节误差和原生奖励。`completed` 只表示参考执行到末尾，不代表高质量跟踪。
- `rollout_rendering.py`、`render_execution.py`：从保存的物理轨迹生成视频，不重新运行策略。
- `experiment_runner.py`：编排源动作／视频入口及编辑前后执行；真实全链验收尚未完成。

运行 25121712 完成原生参考，但首个 MotionFix 重定向动作在 0.04 秒终止。定位为未校准地面高度，已修复；作业 25122012 的编辑前后动作均完成 199/199 步，退出码 0。原始失败产物保留在 `outputs/visualmimic/integration/execution_source`。

```bash
python -m visualmimic.execute_reference --robot-motion input_robot.npz \
  --checkpoint checkpoint.pt --implementation-commit COMMIT --output new_run
python -m visualmimic.render_execution --run new_run --output new_run/rollout.mp4
```

以下内容描述保留的人体层基础接口；基础接口仅依赖 NumPy，真实后端还需要各模型运行环境。

## 模块

- `contracts.py`：显式动作规格、数组校验、分离的三维／视频推理请求、预测与失败记录。
- `storage.py`：原子写入、内容寻址、读取时哈希验证、路径白名单、无 pickle 的动作序列存储。
- `dataset.py`：标准化记录准备、官方划分保留、重复／跨划分动作审计、三维／视频／标签清单分离、渲染编排。
- `rendering.py`、`reconstruction.py`、`editing.py`：可注入的模型后端 Protocol。构造实例时加载权重，逐样本调用不加载标签。
- `adaptation.py`：适配器 Protocol 与严格同规格适配器；未知转换失败，不补零或读取真值。
- `runner.py`：两条路径、阶段错误、耗时、后端身份和中间产物记录。
- `evaluation.py`：评测后端接口、固定分母覆盖率、失败／缺失记录、两组可比性检查。
- `diagnostics.py`：相同表示和时间采样下逐字段 RMSE；无隐式真值对齐，不是官方编辑指标。

完整契约见 [INTERFACES.md](INTERFACES.md)。

## 调用方式

由调用方显式构造真实后端，注入 Runner：

```python
from visualmimic.adaptation import IdentityAdapter
from visualmimic.contracts import GenerationOptions
from visualmimic.runner import Runner, RunConfig
from visualmimic.storage import ArtifactStore

# 这里的目录必须是已分离的输入目录，不应包含标签。
store = ArtifactStore('outputs/visualmimic/run', allowed_read_roots=['prepared/video'])
config = RunConfig(GenerationOptions(
    seed=42,
    duration_policy_id='backend-defined-source-duration',
    sampling_config_id='versioned-backend-sampling-config',
))
# editor / reconstructor 必须实现对应 Protocol，不能省略真实后端配置。
runner = Runner(store, IdentityAdapter(store), editor, reconstructor)
result = runner.run_video(video_request, config)
store.put_json(result)  # 返回可保存到运行索引的 ArtifactRef
```

A 路使用只允许读取三维源输入的 store，再调用 `run_source(source_request, config)`。B 路后端和适配器的输出 store 应与 Runner 输出目录一致，或者加入仅包含这些产物的读取白名单。后端配置哈希应包含完整配置；编辑器配置必须包含文本处理和训练集归一化统计身份。`seed` 由后端负责应用，Runner 只传递并记录，不假装保证第三方 GPU 算子确定性。

评测是独立调用：`evaluate(results, references, protocol, backend, evaluation_store)`。每次评测对应一个条件、一个种子；多种子分别运行。协议固定有序候选集、评测权重和生成／聚合规则。`compare_reports(a, b)` 检查协议、编辑设置及对应指令，并给出共同成功样本。如果两组成功集合不同，需要单独在共同子集重新调用指标后端；该函数不会把不同集合上的分数当作直接对照。原始全量覆盖率报告必须保留。

## 存储与隔离限制

建议分别使用 `prepared/source/`、`prepared/video/`、`prepared/labels/` 以及运行输出目录。视频输入清单仅含视频、时间戳、尺寸、文本和样本身份。渲染相机真值及目标动作只能留在准备／评测侧。

ArtifactStore 的路径白名单会阻止通过其 API 跨目录读取，也检查符号链接解析后的路径。但是普通 Python 后端仍可以自行访问文件系统：这不是 OS 级安全沙箱。实际部署应使用独立容器／作业，只挂载对应输入、模型和输出目录，不挂载标签或真实源动作。当前架构尚未实现容器启动器，不能宣称已验证进程级隔离。

动作 JSON 是第一版交换格式，强调可检查性；大规模数据接入时可增加 NPZ 等后端，同时保留哈希和 schema 校验。当前不复用预测缓存，每次运行都实际调用后端；产物按内容去重。不得据此声称已有断点恢复或模型调用去重功能。

## 验证

在仓库根目录执行：

```sh
python3 -m unittest discover -s tests -p 'test_visualmimic.py' -v
```

测试后端仅定义在测试文件里，标记 `fixture=True`，Runner 默认拒绝运行，测试需显式开启。测试覆盖两条路径共享编辑器、故障传播、错误动作规格、哈希损坏、跨划分动作、标签字段泄漏、路径越界、失败／缺失分母和评测可比性。这些测试验证架构，不验证动作生成质量。

## 尚需接入

1. MotionFix 官方格式读取器、编辑 checkpoint 与官方特征编码／解码。
2. 实际人体视频渲染器与三维重建器，以及经验证的表示转换。
3. 官方评测实现、评测权重和候选集构造。
4. 部署侧的输入目录挂载隔离。

当前不选择这些后端，也不自动下载模型、调用付费 API 或提交训练作业。

## 真实模型与视频准备入口

`prepare_robot_pair` 读取仅含 `source`（ArtifactRef）、`instruction`、`sample_id` 的 JSON。不会读取目标动作。新输出目录必须不存在，以免混淆重跑和失败恢复。

```bash
python -m visualmimic.prepare_robot_pair \
  --input source_manifest.json --source-artifacts source_artifacts \
  --motionfix-repository visualmimic/external/motionfix \
  --gmr-repository visualmimic/external/GMR \
  --checkpoint visualmimic/assets/tmed/checkpoints/last.ckpt \
  --body-models visualmimic/assets/body_models \
  --clip-config visualmimic/assets/clip-vit-large-patch14 --output new_pair
python -m visualmimic.render_source \
  --input source_manifest.json --source-artifacts source_artifacts \
  --body-models visualmimic/assets/body_models --output new_video
```

视频准备结果中的 `video/` 可单独提供给重建进程；`preparation/` 含源动作引用和真实渲染相机，不能提供给视频推理。渲染采用固定相机、真实 SMPL-H 网格、30 fps；本机离屏 OpenGL 需要沙箱外图形访问。

已验证样本 000004：正式 `prepare_robot_pair` 入口与之前独立调用得到完全相同的编辑动作哈希、根位置和关节角；这仅证明该环境下该案例种子 42 的重复一致性。源视频 `outputs/visualmimic/integration/source_human.mp4` 为 120 帧、640×480、30 fps，已检查首／中／末帧。

`backends/gvhmr_reconstructor.py` 提供独立进程的视频入口：严格检查编码帧率／帧数、官方 GVHMR 推理、y-up 到 z-up、固定中性 SMPL-H 表示重建。坐标转换有测试；真实 GVHMR 推理及正式视频入口已在 ICE 单例验证，详见下方匹配实验。缺少官方 SMPL-X 资产时明确失败。


## 单例执行验收结果（25122012）

样本 000004，指令 `don't stand in the beginning`。原动作与编辑动作使用相同冻结 checkpoint 和执行协议，均完成 199/199 步。原动作根位置 RMSE 为 0.198 m，编辑动作为 0.588 m；原生归一化奖励分别为 0.789 和 0.605。这证明三维入口的编辑→重定向→真实仿真→轨迹／视频已在单例上贯通，不证明编辑质量提升或完整第一阶段完成。

结果在 `outputs/visualmimic/execution_pair_25122012/comparison.json`，每组保存 `result.json`、实际状态／动作和视频。初版固定相机视频存在出画问题；`rollout_follow.mp4` 用相同模型和保存的状态重渲染，跟随相机仅改变画面，不重新执行策略。

官方 TMR 运动编码器已严格加载。`tmr_parity.json` 记录了本例特征与官方 MotionFix loader 逐值一致、VAE 均值编码与官方 TEMOS.encode 差异为零。原动作／目标相似度 0.727，编辑动作／目标相似度 0.634；只作单例人体层诊断，不能据此断言机器人指令成功或失败。固定候选集检索支持代码已加入，但尚未完成真实完整候选集评测。

## 官方测试集与固定候选集评测

```bash
python -m visualmimic.prepare_motionfix \
  --dataset visualmimic/assets/motionfix/motionfix_test.pth.tar \
  --splits visualmimic/assets/motionfix/splits.json \
  --annotations visualmimic/assets/motionfix/amt_motionfix_latest.json \
  --output prepared_test
python -m visualmimic.evaluate_human --prepared prepared_test \
  --predictions predictions.json --prediction-artifacts prediction_artifacts \
  --repository visualmimic/external/motionfix --tmr-assets visualmimic/assets/tmr \
  --output human_evaluation
```

预测清单为 `{key, motion}` 行列表，motion 为 canonical MotionSequence 的 ArtifactRef。评测固定使用导入清单顺序；缺失预测时记录完整分母和缺失键，保留 `metrics=null`，不会把成功子集当成更小候选集。`--unedited-baseline` 可以替代 `--predictions`，只用于诊断原动作。

已实际导入官方测试集 1,013 项，结果在 `outputs/visualmimic/prepared_motionfix_test_v3`。完整未编辑诊断评测在 `outputs/visualmimic/tmr_unedited_test1013/report.json`，覆盖率 100%，target→source 的 R@1 为 18.26%、R@5 为 51.33%、中位排名为 5。它不是 TMED 的生成结果，也不是机器人指令成功率；完整 A 编辑结果的固定候选集评测已在 ICE 完成，见下方 6561817 更新；B 完整评测仍未完成。

官方注释的精确 AMASS 路径＋时间片段审计发现 1,177 个片段跨官方划分出现；训练样本 006722 缺少注释，测试集注释齐全。保留官方划分供复现，详见 `official_split_audit.json`；不能声称该协议满足严格动作级互斥。新增视频版本仍必须跟随其原始样本划分，不得随机重新划分。

## 视频重建入口（ICE 单例已验证）

```bash
python -m visualmimic.reconstruct_video \
  --input new_video/video/manifest.json --video-artifacts new_video/video/artifacts \
  --dataset-version VERSION --split test \
  --repository visualmimic/external/GVHMR --python /path/to/gvhmr/python \
  --body-models visualmimic/assets/body_models --output reconstruction
```

输入必须只有视频、时间戳、尺寸、指令和样本 ID。输出 `source_manifest.json` 和 `artifacts/` 可接到同一个 `prepare_robot_pair`，随后同样编译、执行、渲染。正式入口已在 ICE 作业 6560631 验证，随后经过同一编辑器、GMR、参考编译与 SuperMimic 实际执行。单例贯通不代表完整测试集评测已完成。

## ICE 部署要求（用户最新约束）

后续模型、数据、运行环境、产物和计算统一放在 ICE。本地全量编辑已停止：2 项完成，1 项中断；中断记录保留，不能计为生成成功。Rorqual 的已完成单例结果作为历史证据迁移，不再在那里提交新任务。

`deployment/ice_migrate.sh` 迁移代码、外部仓库、全部已下载资产、现有产物与冻结策略；不删除源文件，不提交作业，最后做校验和比较。默认目标是 `/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025`。2026-10-25 ICE 连接恢复，迁移已启动但尚未完成校验。SMPL-X 官网 v1.1 压缩包已获用户授权下载，ZIP 检查通过，SHA-256 为 `cb593838a5d602395735081c7a8fd1d6b04fba261042b84e24644d875039be61`；ICE 上传、完整压缩包校验和中性模型提取已完成，记录为 `licensed-assets/smplx-download.json`。模型 SHA-256 为 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`。

ICE `.venv_gvhmr` 使用 Python 3.10.10、Torch 2.3.0+cu121、Torchvision 0.18.0+cu121、Lightning 2.3.0。官方依赖清单未限制 setuptools 上界，安装后需固定到 75.8.0，以保留旧 Lightning 使用的 `pkg_resources`。`patches/gvhmr-headless-import.patch` 仅删除未使用的 `turtle.forward` 导入，避免服务器要求 Tk；不改变模型计算。核心依赖、完整 demo 导入和实际 GPU 推理均已验证。

Google Drive 主权重链接当时触发下载配额限制，改用官方 README 链接的 Hugging Face 演示所引用的 `camenduru/GVHMR` 镜像。四个必需权重已直接下载到 ICE `gvhmr-checkpoints/`，并逐一匹配镜像公布的 LFS SHA-256。来源元数据和实际哈希在 ICE `logs/gvhmr-mirror-metadata.json` 与 `logs/gvhmr-mirror-sha256.txt`；没有独立证明镜像与当前 Google Drive 文件逐字节相同。GPU 预处理作业 6559503 已 COMPLETED（0:0），耗时 85 秒。人物框、17 点姿态与 1024 维特征均覆盖 120/120 帧，数值全部有限；验证报告在 ICE `runs/video000004/source_human/preprocess_validation.json`。后续作业 6559980 完成三维重建，6560631 完成正式视频入口及 canonical SMPL-H 转换；最终 B 路执行证据见下表。

归档保留原始产物及原路径 provenance。运行前需在 ICE 用已迁移的官方数据重新执行 `prepare_motionfix`，生成 ICE 原生路径清单；不能直接把本机绝对路径当成远程可访问路径。各模型环境须在 ICE 重建，不能复制 macOS 虚拟环境到 Linux。


## ICE 最终匹配单例（6561095，2026-10-25）

样本 000004，官方 test 划分，指令 `don't stand in the beginning`。A 与 B 的编辑均重新在 ICE 同一 `.venv_editor`、CUDA/H100、TMED checkpoint、seed 42 下生成。四组使用相同 G1 模型、冻结 SuperMimic checkpoint、执行协议和 seed 0。作业 COMPLETED、退出码 0:0 表示实验程序完成；其中一条轨迹物理提前终止。

| 条件 | 实际完成步数 | 结果 | 已执行部分根位置 RMSE（m） | 全计划时域归一化回报 |
|---|---:|---|---:|---:|
| A 三维源动作 | 199/199 | completed | 0.1981 | 0.7891 |
| A 三维编辑动作 | 30/199 | terminated，0.6 s | 0.2093 | 0.0692 |
| B 视频重建源动作 | 199/199 | completed | 0.3178 | 0.7815 |
| B 视频重建编辑动作 | 199/199 | completed | 0.2103 | 0.8665 |

A 编辑动作只覆盖 15.08% 计划步数，其短轨迹 RMSE 不能作为完整跟踪成功证据。视频由保存的实际仿真状态生成，四组首／中／末帧已检查；未因失败调整种子或阈值。原生执行初始化物理状态为参考第 0 帧，首个控制目标为第 1 帧；片段结束检查允许最后一帧保持一步，使用同一原生协议。

结果根目录为 ICE `/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025/runs/matched_execution_6561095`。`comparison.json` 汇总四组，每组保留 `result.json`、内容哈希校验的状态／动作、事件和 `rollout.mp4`。策略 SHA-256 为 `387e32872046d7e7a618322dc9c832e1f021cf2aa389c3407143718075bed1e2`；协议 ID 为 `7bebda9fa213cd6ca9c49d9732f382aff21d800f4f2d9ecbdf300969b98dcb94`。较早的 6560903 使用历史 CPU 生成的 A 编辑动作，仅用于运行环境验证，不用于最终匹配对比。

分阶段诊断在同一 ICE 根目录 `runs/single_case_diagnostics_v1/report.json`：重建局部身体旋转误差均值 0.1874 rad、95 分位 0.4209 rad；官方 TMR 对目标动作余弦相似度依次为 A 源 0.7268、A 编辑 0.6796、B 源 0.8072、B 编辑 0.2372。四组重定向结果均无关节限位越界。B 编辑虽然跟踪回报较高，人体目标相似度却降低，不能把跟踪回报等同于指令成功。该任意文本指令的机器人完成度仍为 `unassessed`。

完整阶段一尚未验收：A 官方 1,013 项 GPU 编辑及固定候选集评测作业 6561817 已完成；完整视频化数据、B 全量匹配评测、机器人层指令评价及完整迁移校验仍需完成。上述单例不是完整 benchmark 结果。


终止诊断工具 `python -m visualmimic.termination_diagnostics --run RUN --output NEW_JSON` 校验现有轨迹哈希，不重新执行策略。A 编辑组末状态高度误差 0.248350 m、倾斜误差 0.002848，保存状态未越阈值。实际 mjlab 1.6.0 在刷新运动学前判断终止，读取的身体姿态滞后一个物理子步；现有日志没有记录分项触发标志，因此不能把事后计算当作精确终止原因。报告保存在该组 `termination_diagnostics.json`。后续执行适配器新增记录原生 `native_termination_terms`；不改既有实验结果、阈值或策略。


## 视频数据批处理

`batch_video` 沿用源清单中的完整 `SampleKey` 和顺序，不重新划分数据。渲染准备信息独立存放在 `_preparation`，视频推理清单只包含 key、指令、视频引用、时间戳和尺寸。重建产物清单直接兼容 `batch_edit --manifest`；人体目标动作始终由独立评测器读取。

```bash
python -m visualmimic.batch_video render \
  --manifest datasets/motionfix_test_v1/source/manifest.json \
  --input-artifacts datasets/motionfix_test_v1/source/artifacts \
  --body-models visualmimic/assets/body_models --output runs/test_video_v1
python -m visualmimic.batch_video reconstruct \
  --manifest runs/test_video_v1/manifest.json \
  --input-artifacts runs/test_video_v1_artifacts \
  --body-models visualmimic/assets/body_models \
  --repository visualmimic/external/GVHMR --python .venv_gvhmr/bin/python \
  --output runs/test_reconstruction_v1
```

在 ICE 的 `.venv_editor` 环境、具有 EGL 的 GPU 计算节点运行。每一阶段保存 `run.json`、`items/`、`manifest.json`、`failures.json` 与 `status.json`。`--resume` 核对输入、模型／资产身份、运行环境和完成产物哈希；不确定的中断调用必须额外使用 `--retry-interrupted`。失败样本不会自动重跑，也不会从该阶段的 expected 分母中消失。下游 manifest 只包含成功项，因此完整质量评测必须继续使用原始官方 prepared 数据的全部 1,013 个候选和预期样本，不能把成功清单当作新测试集。

两样本真实渲染＋重建验收作业为 6563875；提交本身不表示该批处理已通过真实运行验收。

## 机器人批处理（新增，等待 ICE 验收）

`batch_robot` 直接使用已有编辑预测或视频重建结果，不重新调用编辑器。`prepare_motionfix` 新增 `tasks.json`，只含完整官方样本 key 与指令；B 路用它保持全分母，无需提供真实源三维动作引用。已有导入目录可从 `source/manifest.json` 提取这两个字段生成等价任务清单。

```bash
# 在 .venv_editor 中重定向已保存的 A 编辑结果。
python -m visualmimic.batch_robot retarget \
  --expected-manifest datasets/motionfix_test_v1/tasks.json \
  --manifest runs/a_test1013_gpu_v1/predictions.json \
  --input-artifacts runs/a_test1013_gpu_v1_artifacts \
  --gmr-repository visualmimic/external/GMR --output runs/a_edited_robot_v1
# 在固定 SuperMimic 环境中编译、实际执行并渲染保存状态。
python -m visualmimic.batch_robot execute \
  --expected-manifest datasets/motionfix_test_v1/tasks.json \
  --manifest runs/a_edited_robot_v1/manifest.json \
  --input-artifacts runs/a_edited_robot_v1_artifacts \
  --checkpoint policies/supermimic-lafan1-39100.pt \
  --implementation-commit d9be385b3eeb13a638fb33fb200d3bfd9be9839f \
  --seed 0 --output runs/a_edited_execution_v1
```

A 未编辑对照可把完整 source manifest 作为 expected manifest，省略 retarget 的 `--manifest`。B 未编辑对照使用完整 task manifest，并把 reconstruction 的 source manifest 传给 `--manifest`；B 编辑对照传其 `batch_edit` predictions。每次指定相应产物目录，不把真实源动作或目标标签目录授权给 B 的产物读取器。

每个批次共享内容寻址产物库和冻结策略文件；每个样本仍独立重置原生环境、使用相同执行 seed。GMR 每段建立新实例，避免跨片段状态污染。缺失预测保留为失败条目，不能通过缩小清单提高完成率。`outcomes.json` 分开统计 completed、terminated、truncated 和 failed_or_missing；程序成功不等于机器人完成动作，更不等于指令成功。原生终止标志与实际状态／动作保留在每个 result 的引用中。新增入口尚未在 ICE 实际验收，不作为已有实验结果。


### 2026-10-25 连接中断后的恢复顺序

ICE SSH 连续超时；完整迁移进程已明确退出（255，broken pipe），不能声称迁移完成。6561817（全量 A 编辑）和 6563875（两样本视频批处理）最后可见状态为 PENDING，断线后的状态未知。不要因无法观察而重新提交这些作业。

恢复连接后先查询两个原作业及其日志／产物，再同步尚未部署的 `batch_ledger.py`、重构后的 `batch_video.py`、`batch_robot.py`、新增任务清单的 `prepare_motionfix.py` 和对应测试。运行 ICE 回归检查，再提交已准备的机器人批处理两样本验收脚本。新机器人入口目前仅通过静态语法检查，没有真实运行验收证据。已有 28 项阶段一测试和先前 4 项视频批处理测试的通过记录，不覆盖断线后这些改动。

迁移可在确认旧进程退出后重启 `deployment/ice_migrate.sh`，保留 partial 文件续传；不得删除现有远端运行产物。最后检查迁移校验输出并同步最新代码，生成 ICE 原生路径清单。以上依赖完成前，不提交完整 B 分支批量实验，也不将阶段一标记完成。


### 2026-10-25 SSH 恢复后的实测更新

已重新连接 ICE，核对原作业而未重复提交。6561817 为 COMPLETED、退出码 0:0、耗时 27 分 27 秒。生成状态为 expected=1,013、succeeded=1,013、failed=0；`runs/a_test1013_gpu_v1_evaluation/report.json` 为 evaluated、available=1,013、coverage=1.0。官方 TMR 固定候选检索结果：R@1=12.54%、R@2=23.59%、R@3=30.50%、R@5=40.77%、R@10=54.29%、MedR=8、AvgR=57.48。该结果是人体层动作检索指标，不是机器人执行／指令完成率，也不构成 B 分支结果。

6563875 视频批处理验收失败：两个样本均在渲染阶段报 `(standalone) XOpenDisplay: cannot open display`，未生成可重建视频。Linux 无显示器节点需显式选择 ModernGL EGL；单设 PYOPENGL_PLATFORM 不会更改 ModernGL 的默认 X11 选择。`human_renderer` 已修复构造过程，原失败记录保留，需新作业验证修复。

迁移已从已确认退出的旧进程续传；最终校验仍待完成。断线期间新增的批处理代码已部署到 ICE，正在进行回归与真实执行验收。


## 全量 B 实验已提交（2026-10-25）

ICE 实验目录为 `campaigns/b_test1013_v1`，覆盖官方 test 全部 1,013 项。数据按官方顺序步进划成 64 片，合并时恢复官方顺序，不重新划分样本。`plan.json` 固定输入身份；`jobs.json` 记录每次提交的作业 ID，防止网络重连后重复提交。

队列每用户最多允许 50 个已提交任务，因此 64 片视频任务装入 32 个数组元素，每个处理两片，最多同时使用 4 张 H100。首轮 64 元素提交被 QoS 明确拒绝，没有启动，拒绝记录保留在 `submission_attempt1.json`。成功视频数组为 6589463，自动等待其结束的编辑＋TMR 评测为 6589464。两组 B 机器人执行与汇总的实际 ID 见 `jobs.json`。

完整顺序：源动作渲染 → 仅视频 GVHMR 重建 → 汇总并验证哈希 → 与 A 相同 TMED checkpoint／CUDA／seed 42 编辑 → 原始 1,013 候选集 TMR 评测 → B 未编辑和编辑两组分别 GMR、冻结 SuperMimic／seed 0、实际轨迹视频 → 全分母机器人结果汇总。机器人任务使用 16 个数组元素，每个处理四个数据分片，最多并发 4 张 GPU。对应可复现脚本为 `deployment/b_video_array.sbatch`、`b_edit_evaluate.sbatch`、`b_robot_array.sbatch`、`b_robot_summary.sbatch`。

重建失败、未执行和中断样本不会被当成成功，也不会从全分母中移除。若缺预测，完整固定候选检索指标保持 null；机器人汇总分别报告 completed／terminated／truncated／failed_or_missing。指令成功不由跟踪完成度代替。分片与合并相关 7 项检查已在 ICE 通过；提交不等于已运行或已完成。A 全量机器人对照尚需另行衔接，不能把 B 作业提交当成整个阶段一完成。
