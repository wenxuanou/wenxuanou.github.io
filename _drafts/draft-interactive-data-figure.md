---
title: "大模型参数量一览：按参数量轴排布的交互图"
date: 2026-11-18
categories:
  - blog
tags:
  - 数据可视化
  - 大模型
  - Jekyll
layout: single
author_profile: true
excerpt: "把模型按参数量排在对数轴上：柱高统一，MoE 用深浅两色区分激活与总参数量，轴下方按应用场景分组。"
---

这张图借用了时间轴的排布方式，只是横轴换成了**模型参数量**（对数刻度，否则 2.6B 和 1T 挤在一起没法看）。每个模型一根柱子，鼠标移到柱子上会放大高亮，并在柱顶上方浮出详细信息。

{% include components/data-figure.html
   id="llm-params"
   src="/assets/data/llm-params.csv"
   type="timeline"
   x="total_params"
   y="active_params"
   series="scenario"
   label="model"
   title="大语言模型参数量分布"
   xlabel="参数量（对数刻度）"
   height="470"
   caption="x 轴为参数量；MoE 模型用两段深浅表示激活参数与总参数量；轴下方横条是应用场景分组。" 
%}

几个约定：

- **柱高是统一的**。横轴已经编码了参数量，柱高不再重复编码，这样才像时间轴而不是柱状图。
- **MoE 模型**用同色相的两段表示：深色是激活参数量，浅色补满到总参数量，所以深色占比就是激活比例（例如 DeepSeek-V3 是 37B / 671B ≈ 5.5%）。
- **轴下方的横条**按应用场景把模型框在一起，同一场景的柱子颜色相同。
- 悬停时柱子会放大并浮现详情：模型名、场景、总参数量、激活参数量与比例，以及表格里其余列（机构、发布时间、备注）。
- **模型多了可以横向滚动**：画布会按模型数量和最小柱间距自动加宽，超出容器宽度的部分左右滚动查看（鼠标滚轮、触控板横滑、手机滑动都可以）。参数量的上下限由数据算出，后面加入更大的模型时轴会自动延伸。

## 在你自己帖子里怎么用

1. 准备一个 CSV 放进 `assets/data/`，**保存为 UTF-8**（Excel：另存为 →「CSV UTF-8（逗号分隔）」）。需要的列：

   | 列 | 必需 | 说明 |
   |---|---|---|
   | 模型名 | 是 | `label="model"` 指定 |
   | 总参数量 | 是 | 单位为**十亿**，`x="total_params"` 指定 |
   | 激活参数量 | 否 | 只有 MoE 填，`y="active_params"` 指定；留空表示稠密模型 |
   | 应用场景 | 否 | `series="scenario"` 指定，决定轴下方的分组条 |
   | 其他列 | 否 | 机构、时间、备注等会原样出现在悬停详情里 |

2. 在帖子里顶格写一行 include（不要缩进，也不要放进代码块）：

{% raw %}
```markdown
{% include components/data-figure.html
   id="llm-params"
   src="/assets/data/llm-params.csv"
   type="timeline"
   x="total_params"
   y="active_params"
   series="scenario"
   label="model"
   title="大语言模型参数量分布"
   xlabel="参数量（对数刻度）"
   caption="数据来源：内部整理。" %}
```
{% endraw %}

3. 提交后由 GitHub Actions 构建，图表在浏览器端读取 CSV 绘制，改数据只需要换 CSV。

## 注意事项

- 组件**零外部依赖**：图表由页面内的 SVG 绘制，不加载 Plotly 或任何 CDN 脚本，国内网络、离线预览都不受影响。
- 参数量单位是**十亿（B）**；`1000` 会显示成 `1T`。轴的范围（`1B`~`10T` 之类）由数据里最小/最大的模型算出，加新模型不用改代码。
- 参数量完全相同的两个模型会画在同一位置、互相遮挡，建议在数据里合并或错开记录。
- 样例数据是公开资料整理的近似值，仅用于演示；换成你自己表格时记得核对。
- 数据文件由 GitHub Pages 静态托管，默认约 10 分钟缓存，更新后不会立刻反映。
