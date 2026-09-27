# PsychEthicsEval Phase Two Core

这是 PsychEthicsEval 的 Phase Two 核心流水线：读取题目，生成 MCQ 或 OEQ 请求，调用固定模型，保存不可变执行记录，自动审计并导出带 `prediction` 的原始数据文件。

仓库目标是保留“输入 → 处理 → 输出”的完整框架，不保留 Judge、旧实验轮次、网页界面、历史报告或中间调试工具。

## 最终运行链

```text
mcq_phase_2.json ──> B1 prompt ───────────────────────────> Answer model ──> MCQ parser ─┐
                                                                                         ├─> audit ─> submission.zip
oeq_phase_2.json ──> identity v13 ─> permission boundary ─> RAG ─> Answer model ─────────┘
```

- MCQ：固定 B1 Prompt，严格解析零起始选项索引数组。
- OEQ：v13 身份分类；无法可靠绑定到题目原文时自动弃权，不授予角色特定知识权限。
- RAG：固定澳大利亚语境知识单元，使用本地 MiniLM 语义检索；模型不能自行扩大可访问分区。
- 执行：有界并发、RPM/TPM 限制、传输重试上限、SQLite 追加式事件链。
- 导出：只有完整且未失败的审计结果可以生成两个原文件名和 `submission.zip`。
- Judge：最终 Phase Two 路径不调用 Judge；运行清单固定记录 `judge: 0`。

## 关于 5B Freeze

最终提交并不是由一个 Freeze 从头生成的单次运行：

- A222 在 `full-5b-v5` 下完成主体运行并返回 3,747 条答案；
- 其中 42 条 MCQ 可以证明从未发起模型调用；
- A223 在 `full-5b-v6` 下使用修订后的 MCQ 包装解析器和补全逻辑运行这 42 条；
- 最终导出由 A222 结果和 A223 补全结果组成，并经 v6 审计与导出。

因此，v6 是最终补全、合并、审计与导出所用版本，但不是全部 3,789 条预测唯一的从头运行版本。历史冻结包应作为发布证据保存，不应被描述成核心源码本身。

## 精简后的目录

```text
.
├── README.md
├── pyproject.toml
├── config/
│   ├── datasets.phase2.json
│   └── au_resource_pack/RAG-IDENTITY-KNOWLEDGE-v3.json
├── prompts/PHASE2_PROMPTS.md
├── scripts/run_full_phase2.py
├── src/psychethicseval_lab/
│   ├── api_client.py
│   ├── contracts.py
│   ├── identity_classifier_v8.py       # v12 仍使用的候选提取基础
│   ├── identity_classifier_v12.py      # v13 仍使用的严格证据绑定器
│   ├── identity_classifier_v13.py      # 最终身份分类策略
│   ├── identity_boundary_v3.py
│   ├── domain_retrieval_v3.py
│   ├── domain_retrieval_v4.py
│   ├── knowledge_v3.py
│   ├── minilm_retriever.py
│   ├── oeq_materialization_v3.py
│   ├── mcq_prediction_v2.py            # v3 仍使用的严格基础解析器
│   ├── mcq_prediction_v3.py
│   ├── full_run_manifest.py
│   └── full_execution.py
├── artifacts/semantic_embeddings/      # 固定 MiniLM 模型、运行时和许可证
└── tests/test_phase2_core.py
```

保留 v8/v12 和 MCQ v2 不是保留旧产品路线，而是因为最终模块明确复用了这些经过验证的底层契约。

## 数据

默认读取仓库同级目录：

```text
../dataSET/mcq_phase_2.json
../dataSET/oeq_phase_2.json
```

文件数量和 SHA-256 必须与 `config/datasets.phase2.json` 一致。代码拒绝已经包含 `prediction`、MCQ 答案键或 OEQ `inquirer` 标签的输入。

## 验证

验证不会调用远程模型：

```bash
PYTHONPATH=src python3 -m unittest -v tests.test_phase2_core
```

核心测试覆盖：v13 自动弃权与权限边界、RAG 请求生成、MCQ 解析、输入到输出的小型完整执行、事件链重放、3,789 条全量 manifest，以及凭据元数据不进入 manifest。

## 运行

先生成并验证运行清单：

```bash
PYTHONPATH=src python3 scripts/run_full_phase2.py prepare \
  --manifest var/phase2-manifest.json \
  --checkpoint var/phase2-checkpoint.sqlite3

PYTHONPATH=src python3 scripts/run_full_phase2.py verify \
  --manifest var/phase2-manifest.json
```

执行是显式付费操作。凭据只在 `execute` 开始时从进程环境读取：

```bash
PHASE2_IDENTITY_CREDENTIAL='...' \
PHASE2_ANSWER_CREDENTIAL='...' \
PYTHONPATH=src python3 scripts/run_full_phase2.py execute \
  --manifest var/phase2-manifest.json \
  --report var/phase2-audit.json
```

审计和导出不需要凭据：

```bash
PYTHONPATH=src python3 scripts/run_full_phase2.py audit \
  --manifest var/phase2-manifest.json \
  --report var/phase2-audit.json

PYTHONPATH=src python3 scripts/run_full_phase2.py export \
  --manifest var/phase2-manifest.json \
  --output output/phase2-submission
```

已有不完整或不确定的检查点不会自动续跑，避免把无法证明来源的结果混入提交。

## 凭据与安全

- 仓库不保存真实密钥、`.env`、Keychain 服务名或账户名。
- manifest、SQLite、错误事实和导出中不记录凭据或凭据环境变量名。
- 只有执行适配器知道两个通用环境变量；凭据不被打印或序列化。
- HTTP 错误正文和异常消息不进入失败记录，避免意外泄漏请求信息。

发布前仍应对当前工作树和新 Git 历史执行密钥扫描。若从旧提交历史直接发布，旧的非秘密配置名称仍可能存在于历史对象中；建议验收后从精简树创建新的干净发布仓库，而不是公开包含实验历史的旧仓库。

## 环境限制

- Python 3.11+；当前固定语义运行时为 macOS Apple Silicon / CPython 3.12。
- `artifacts/semantic_embeddings/` 缺少清单声明的模型、许可证、wheel 或展开运行时时会失败关闭。
- 项目不声称 Phase Two accuracy，因为 Phase Two 输入不包含本地真值。
- 自动审计验证来源、结构与覆盖率，不代表模型答案必然符合临床、法律或伦理事实。

## License

仓库尚未声明统一许可证。MiniLM 的许可证位于 `artifacts/semantic_embeddings/minilm-l6-v2/LICENSE`。公开发布前请确认代码、数据集和第三方资产的再分发权限。
