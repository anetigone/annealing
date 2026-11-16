# CVPR 2026 与 2027 模板比较

核对日期：2026-11-02。以根目录两份 author-kit 的逐文件 diff 为本地依据，另核对[官方发布页](https://github.com/cvpr-org/author-kit/releases)和[2027 作者指南](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)。本地目录与官方最新 tag 的所有文件是否完全一致尚未核验，以下不混称二者。

| 项目 | 本地 2026 → 2027 | 写稿影响 |
|---|---|---|
| 匿名标题栏 | 移除 Paper ID | 不再定义或打印 paperID |
| 页顶左右角 | 仅保留会议名，删除编号 | 沿用 2027 cvpr.sty |
| 保密页眉 | 删除 Submission 后的编号 | 不自行加回编号 |
| 年份 | 2026 → 2027 | main 和 rebuttal 均更新 |
| 使用说明 | 删除填写 Paper ID 的段落 | 无需填写占位星号 |
| 版式 | 仍为 10pt、双栏、letterpaper | 无需改页边距和字号 |
| 参考文献及示例 | bst、bib、preamble、摘要、排版说明、finalcopy 和补充示例均相同 | 无实质迁移成本 |
| 打包辅助文件 | 本地 2027 缺少 2026 的 .github workflow、.gitignore | 属于本地打包差异，不能据此认定官方取消 CI |

官方 2027 release 还列出 teaser/strip/preamble 调整和 CI Node 24 升级；这些不是本地两份文件 diff 的变化，不能声称已在用户这份包中体现。

## 模板之外但影响正文的要求

2027 指南明确要求所有投稿附匿名 LLM Usage Statement，说明工具名、涉及部分与使用程度；即使未使用也须声明。正文最多 8 页，参考文献与该声明不计入。用户这份本地 2027 示例没有该声明，不能只替换示例文字就认为已满足新规则。新稿已加入本次 Codex 辅助整理、翻译、LaTeX 和检索的披露，全部研究阶段的完整披露仍需作者核实。

指南还增加 reciprocal-reviewer quota；这是投稿资格政策，不是 LaTeX 样式差异。正式投稿前应按官网当时版本再核对政策。
