# med-rag-lab

一个面向**中文临床指南**的检索增强问答实验项目：把公开指南 PDF 解析成可引用的
chunk，比较 BM25 / 向量 / 混合 / 重排四组检索方案，并在回答层做**引用约束**与
**低分拒答**。

语料全部来自公开指南原文，评测集为人工标注的 26 道题，所有指标可复现。

## 0. 快速开始

```bash
# 1. 依赖
uv sync

# 2. 语料与模型（模型不入库，需单独下载）
bash scripts/download_data.sh
bash scripts/download_models.sh

# 3. 重建语料与评测集
python src/ingest/parse.py
python eval/build_eval.py

# 4. 跑一组检索
python scripts/search.py --retriever hybrid_rerank "高血压诊断标准 140 90"

# 5. 起服务
uvicorn api.app:app --app-dir src --port 8000
```

> 模型下载脚本会把 ONNX 文件放到 `models/`，代码优先使用本地目录；目录不存在时
> 才回退到 fastembed 的在线下载。部分环境 Python 默认 CA 证书不可用，脚本已显式
> 指定系统证书路径。

## 1. 语料

| 文档 | 版本 | 页数 | chunk 数 |
| --- | --- | --- | --- |
| 中国高血压防治指南 | 2024 年修订版 | 98 | 435 |
| 中国 2 型糖尿病防治指南 | 2020 年版 | 95 | 398 |

- 解析：PyMuPDF 抽取文本 → 去页眉页脚/重复行 → 按中文句末标点合并断行 →
  450 字窗口、80 字重叠切分。
- 每个 chunk 保留 `doc_id` 与 `page`，因此答案可以精确引用到“指南 + 页码”。

## 2. 检索链路

```
问题
 ├─ BM25（jieba 分词）────────┐
 └─ BGE-small-zh 向量召回 ────┴─ RRF 融合 Top10 ─ 交叉编码器重排 ─ Top5
```

- 向量模型：`BAAI/bge-small-zh-v1.5`（ONNX，512 维，本地推理）
- 重排模型：`BAAI/bge-reranker-v2-m3-int8`（ONNX INT8）
- 融合：Reciprocal Rank Fusion，`k=60`，两路各取 Top50
- 没有 GPU 也能跑；语料只有 833 个 chunk，向量检索用归一化矩阵点积即可。

## 3. 评测

评测集：`eval/eval_set.jsonl`，26 题（高血压 14 / 糖尿病 12）。每题标注一个
`gold_phrase`，构建时自动映射到包含该短语的**最短 chunk** 作为唯一正确项。

运行：

```bash
python eval/build_eval.py
python eval/run_eval.py --retriever bm25
python eval/run_eval.py --retriever dense
python eval/run_eval.py --retriever hybrid
python eval/run_eval.py --retriever hybrid_rerank
```

### 两套口径

指南会把同一条推荐同时写进要点框、正文、专章和另一本指南，所以“正确答案”常常
不止一个 chunk。只按单一 `gold_chunk_id` 判分会低估检索效果，因此同时报两套指标：

- **严格口径**：top-k 是否包含标注的那一个 `gold_chunk_id`；
- **短语覆盖**：top-k 里是否有任意 chunk 包含答案句 `gold_phrase`（能容忍切分重叠
  和原文重复，但不能容忍换一种说法）。

严格口径：

| 方案 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.2692 | 0.5385 | 0.8077 | 0.3947 | 0.4901 |
| 向量（bge-small-zh） | 0.0385 | 0.5769 | 0.6923 | 0.2029 | 0.3186 |
| 混合 RRF | 0.0769 | 0.8077 | 0.9231 | 0.3302 | 0.4728 |
| 混合 + 重排 | **0.3846** | 0.7692 | 0.9231 | **0.5280** | **0.6212** |

短语覆盖口径：

| 方案 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- |
| BM25 | 0.5385 | 0.8077 | 0.4203 |
| 向量（bge-small-zh） | 0.6538 | 0.7308 | 0.2667 |
| 混合 RRF | **0.8462** | 0.9231 | 0.3920 |
| 混合 + 重排 | 0.7692 | 0.9231 | **0.5761** |

两套口径的差异集中在 h13 这类题：糖尿病指南写“一般糖尿病合并高血压 <130/80”，
高血压指南写“有合并症者 <130/80”，意思相同但不是同一句原文。严格口径判错，
人工看却是对的。这也是为什么需要多 gold 或关键值覆盖，单一口径一定会失真。

结论（基于当前数据）：

- 单路 BM25 的首位命中明显好于小向量模型，但 Top5 覆盖不足；
- 混合检索把 Recall@5 从 0.54 提到 0.81，代价是首位排序被拉低；
- 重排把 Recall@1 从 0.08 提到 0.38、MRR 从 0.33 提到 0.53，但 Top5 反而
  略降到 0.77：它只能在候选池内重排，且会把个别正确项挤出前 5。

### 候选池大小 vs 延迟（重排阶段）

| 候选池 | Recall@1 | Recall@5 | MRR | 单次查询耗时 |
| --- | --- | --- | --- | --- |
| 8 | 0.3846 | 0.7308 | 0.5282 | 2.4 s |
| 10 | 0.3846 | 0.7692 | 0.5280 | 3.1 s |
| 15 | 0.3846 | 0.7692 | 0.5380 | 4.7 s |
| 20 | 0.3462 | 0.7692 | 0.5010 | 9.6 s |

默认取 10：再扩大候选池，指标不涨，CPU 延迟线性上升。这是“效果/成本”取舍，
不是“越大越好”。

## 4. 回答层

`POST /ask`：

```bash
uvicorn api.app:app --app-dir src --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"一般高血压患者的诊室血压目标是多少？"}'
```

- 每个来源编号 `[1]` 对应 `doc_title + page + chunk_id`；
- 重排最高分低于阈值时直接拒答，返回“根据现有资料无法确定”；
- 阈值默认 `2.0`：在 26 道可答题 + 10 道语料外问题上标定，可答题最低分 2.36、
  语料外最高分 1.53，取中间值后两组错误率都是 0（样本小，仅作演示）；
- 配置 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 后走 LLM 生成，并校验
  答案里的引用编号必须落在检索结果范围内，否则按拒答处理；
- 未配置 LLM 时退化为**抽取式**回答，保证服务可离线演示。

## 5. 目录

```
src/ingest/parse.py         PDF 解析、清洗、切分
src/retrieval/bm25.py       BM25
src/retrieval/dense.py      BGE 向量召回
src/retrieval/hybrid.py     RRF 融合
src/retrieval/rerank.py     交叉编码器重排
src/generation/answer.py    引用、拒答、LLM/抽取式回答
src/api/app.py              FastAPI
eval/build_eval.py          gold_phrase -> gold_chunk_id
eval/run_eval.py            消融评测（严格 + 短语覆盖两套口径）
eval/calibrate.py           拒答阈值标定
eval/rank_delta.py          逐题名次变化归因
eval/compare.py             多方案汇总
scripts/search.py           统一检索 CLI
scripts/download_data.sh    下载并校验指南 PDF
scripts/download_models.sh  下载 ONNX 模型
tests/                      拒答与引用校验单测
```

## 6. 已知限制

- 评测集只有 26 题，指标方差较大，结论只对本语料成立；
- `bge-small-zh` 是轻量模型，换更大模型（bge-large-zh / bge-m3）预期还能提升；
- chunk 按字符切分，未做表格结构化，表格类问题的召回偏弱；
- 抽取式回答只保证“引用正确”，不保证“直接回答问题”，LLM 模式才是完整形态。
