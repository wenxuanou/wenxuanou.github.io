/**
 * data-figure.js — 零依赖的交互式图表组件。
 *
 * 为什么不用 Plotly：模板默认从 jsDelivr 按需加载 Plotly，但在国内网络下该请求
 * 经常被卡住（onload/onerror 都不触发），页面会永远停在「正在加载数据…」。
 * 这里改成自己用 SVG 画，不请求任何外部资源：
 *   - 首屏无需等待 4.5MB 的图表库，也没有 CDN 单点故障；
 *   - 交互（hover 数值、图例开关、拖拽缩放、双击复位）由本文件实现。
 *
 * 结构分两层：
 *   buildScene()  纯计算：数据 → 像素几何（可单测，不需要 DOM）
 *   renderSvg()   纯字符串：几何 → SVG 标记（可单测）
 *   DOM 层        只负责挂事件、更新 tooltip、跟随主题与容器尺寸。
 */

const PALETTE = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3", "#ff6692", "#b6e880"];
const STATUS_CLASS = "data-figure__status";
const TOOLTIP_CLASS = "data-figure__tooltip";
const NUMERIC_RE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

const THEMES = {
  light: { ink: "#2a3f5f", muted: "#6a7b8c", grid: "#ebf0f8", axis: "#a2b1c6", bg: "transparent" },
  dark: { ink: "#f2f5fa", muted: "#9aa7b4", grid: "#283442", axis: "#506784", bg: "transparent" },
};

/* ==========================================================================
   纯函数：解析与数据整形
   ========================================================================== */

export function stripBom(text) {
  return String(text === null || text === undefined ? "" : text).replace(/^\uFEFF/, "");
}

/** 单元格文本 → number / string / null。数字判定从严，"1,234" 这类格式化文本保持字符串。 */
export function coerce(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  const text = String(raw).trim();
  if (text === "") {
    return null;
  }
  return NUMERIC_RE.test(text) ? Number(text) : text;
}

/**
 * 解析 CSV（RFC 4180 子集）：引号包裹、引号内逗号与换行、"" 转义、
 * CRLF / LF、UTF-8 BOM、空行、行尾缺列。
 */
export function parseCsv(text, options = {}) {
  const source = stripBom(text);
  const records = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];

    if (inQuotes) {
      if (char === '"') {
        if (source[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }
    if (char === ",") {
      row.push(field);
      field = "";
      continue;
    }
    if (char === "\r" || char === "\n") {
      if (char === "\r" && source[i + 1] === "\n") {
        i += 1;
      }
      row.push(field);
      records.push(row);
      row = [];
      field = "";
      continue;
    }
    field += char;
  }

  if (field !== "" || row.length > 0) {
    row.push(field);
    records.push(row);
  }

  const meaningful = records.filter((cells) => cells.some((cell) => String(cell).trim() !== ""));
  if (meaningful.length === 0) {
    return { columns: [], rows: [] };
  }

  const hasHeader = options.header !== false;
  let columns = hasHeader
    ? meaningful[0].map((cell, index) => {
        const name = String(cell).trim();
        return name === "" ? "col" + (index + 1) : name;
      })
    : meaningful[0].map((_, index) => "col" + (index + 1));
  const dataRecords = hasHeader ? meaningful.slice(1) : meaningful;

  const seen = new Map();
  columns = columns.map((name) => {
    const count = (seen.get(name) || 0) + 1;
    seen.set(name, count);
    return count === 1 ? name : name + "_" + count;
  });

  const rows = dataRecords.map((cells) => {
    const record = {};
    columns.forEach((name, index) => {
      record[name] = coerce(cells[index]);
    });
    return record;
  });

  return { columns, rows };
}

/** 解析「对象数组」形式的 JSON。 */
export function parseJson(text) {
  const data = JSON.parse(stripBom(text));
  if (!Array.isArray(data)) {
    throw new Error("JSON 数据必须是对象数组（[{...}, {...}]）");
  }
  const columns = [];
  data.forEach((item) => {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      Object.keys(item).forEach((key) => {
        if (!columns.includes(key)) {
          columns.push(key);
        }
      });
    }
  });
  const rows = data.map((item) => {
    const record = {};
    columns.forEach((key) => {
      record[key] = coerce(item && typeof item === "object" ? item[key] : undefined);
    });
    return record;
  });
  return { columns, rows };
}

export function isNumericColumn(rows, column) {
  return rows.some((row) => typeof row[column] === "number");
}

export function sortRowsBy(rows, column) {
  return rows.slice().sort((a, b) => {
    const left = a[column];
    const right = b[column];
    if (left === null || left === undefined) {
      return 1;
    }
    if (right === null || right === undefined) {
      return -1;
    }
    if (typeof left === "number" && typeof right === "number") {
      return left - right;
    }
    return String(left).localeCompare(String(right), undefined, { numeric: true });
  });
}

/**
 * 行数据 → 系列数组：[{ name, x: [...], y: [...] }]。
 * 宽表：每个 y 列一个系列；长表（给了 series）：每个分组一个系列。
 */
export function rowsToSeries(rows, spec = {}) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return [];
  }
  const columns = Object.keys(rows[0] || {});
  if (columns.length === 0) {
    throw new Error("数据文件里没有可用的列");
  }

  const xColumn = spec.x || columns[0];
  if (!columns.includes(xColumn)) {
    throw new Error("找不到 x 轴列「" + xColumn + "」；可用列：" + columns.join("、"));
  }

  const seriesColumn = spec.series || "";
  const candidates = columns.filter(
    (column) => column !== xColumn && column !== seriesColumn && isNumericColumn(rows, column)
  );
  let yColumns = Array.isArray(spec.y) ? spec.y.filter(Boolean) : [];
  if (yColumns.length === 0) {
    yColumns = seriesColumn ? candidates.slice(0, 1) : candidates;
  }
  if (yColumns.length === 0) {
    throw new Error("没有找到可绘制的数值列；可用列：" + columns.join("、"));
  }
  yColumns.forEach((column) => {
    if (!columns.includes(column)) {
      throw new Error("找不到 y 轴列「" + column + "」；可用列：" + columns.join("、"));
    }
  });

  const ordered = spec.sort === "x" ? sortRowsBy(rows, xColumn) : rows;

  if (seriesColumn) {
    if (!columns.includes(seriesColumn)) {
      throw new Error("找不到分组列「" + seriesColumn + "」；可用列：" + columns.join("、"));
    }
    const groups = new Map();
    ordered.forEach((row) => {
      const key = row[seriesColumn] === null || row[seriesColumn] === undefined
        ? "(空)"
        : String(row[seriesColumn]);
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(row);
    });
    return Array.from(groups.entries()).map(([name, groupRows]) => ({
      name,
      x: groupRows.map((row) => row[xColumn]),
      y: groupRows.map((row) => row[yColumns[0]]),
    }));
  }

  return yColumns.map((column) => ({
    name: column,
    x: ordered.map((row) => row[xColumn]),
    y: ordered.map((row) => row[column]),
  }));
}

/** 把数值格式化成适合读的短文本。 */
export function formatValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return value === null || value === undefined ? "—" : String(value);
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 1 ? 2 : 3;
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

/* ==========================================================================
   纯函数：几何计算
   ========================================================================== */

/** 生成“好看”的刻度：1/2/2.5/5/10 × 10^n 步长，覆盖 [min, max]。 */
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { ticks: [], step: 1 };
  }
  if (max <= min) {
    max = min + 1;
  }
  const raw = (max - min) / Math.max(count, 1);
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = magnitude * 10;
  for (const multiplier of [1, 2, 2.5, 5, 10]) {
    if (raw <= magnitude * multiplier) {
      step = magnitude * multiplier;
      break;
    }
  }
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 1e-9; value += step) {
    ticks.push(Math.round(value / step) * step);
    if (ticks.length > 50) {
      break;
    }
  }
  if (ticks.length && ticks[ticks.length - 1] < max) {
    ticks.push(Math.round((ticks[ticks.length - 1] + step) / step) * step);
  }
  return { ticks, step };
}

function uniqueCategories(series) {
  const seen = new Set();
  const out = [];
  series.forEach((item) => {
    item.x.forEach((value) => {
      const key = String(value);
      if (!seen.has(key)) {
        seen.add(key);
        out.push({ key, label: String(value).slice(0, 10) });
      }
    });
  });
  return out;
}

/**
 * 数据 → 像素几何。
 *
 * @param {Array} series rowsToSeries() 的输出
 * @param {object} spec   { type, title, xlabel, ylabel, height }
 * @param {object} options { width, theme, hidden:Set<number>, domain:{from,to}|null }
 * @returns {object} scene
 */
export function buildScene(series, spec, options = {}) {
  const width = Math.max(240, Math.round(options.width || 720));
  const height = Math.max(160, Math.round(options.height || spec.height || 420));
  const theme = options.theme || THEMES.light;
  const hidden = options.hidden instanceof Set ? options.hidden : new Set();
  const kind = spec.type === "bar" ? "bar" : spec.type === "area" ? "area" : spec.type === "scatter" ? "scatter" : "line";

  const visibleFlags = series.map((_, index) => !hidden.has(index));
  const visibleSeries = series.filter((_, index) => visibleFlags[index]);

  const categories = uniqueCategories(series);
  const total = categories.length;
  const domain = normalizeDomain(options.domain, total);
  const shownCount = domain.to - domain.from + 1;
  const shown = categories.slice(domain.from, domain.to + 1);

  const margin = { left: 64, right: 24, top: 18, bottom: 40 };
  if (spec.title) margin.top += 26;
  if (spec.xlabel) margin.bottom += 18;
  const showLegend = series.length > 1;
  if (showLegend) margin.bottom += 24;

  const plot = {
    x0: margin.left,
    y0: margin.top,
    x1: width - margin.right,
    y1: height - margin.bottom,
  };
  const plotWidth = Math.max(plot.x1 - plot.x0, 10);
  const plotHeight = Math.max(plot.y1 - plot.y0, 10);

  // x 方向：折线把点铺满绘图区，柱状用等宽槽位中心。
  const positionAt = (shownIndex) => {
    if (kind === "bar") {
      return plot.x0 + ((shownIndex + 0.5) / Math.max(shownCount, 1)) * plotWidth;
    }
    return plot.x0 + (shownIndex / Math.max(shownCount - 1, 1)) * plotWidth;
  };

  // y 方向范围：柱状从 0 起；其余按可见数据上下留白。
  const values = [];
  visibleSeries.forEach((item) => {
    item.y.forEach((value) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        values.push(value);
      }
    });
  });
  const fallbackValues = series.flatMap((item) => item.y.filter((v) => typeof v === "number" && Number.isFinite(v)));
  const basis = values.length ? values : fallbackValues;
  let yMin;
  let yMax;
  if (kind === "bar") {
    yMin = Math.min(0, ...(basis.length ? basis : [0]));
    yMax = Math.max(...(basis.length ? basis : [1]));
    if (yMax === yMin) yMax = yMin + 1;
  } else if (basis.length) {
    yMin = Math.min(...basis);
    yMax = Math.max(...basis);
    const pad = (yMax - yMin) * 0.1 || Math.abs(yMax) * 0.1 || 1;
    yMin -= pad;
    yMax += pad;
  } else {
    yMin = 0;
    yMax = 1;
  }

  const { ticks: yTickValues } = niceTicks(yMin, yMax, 5);
  // 把轴范围对齐到刻度范围：否则最外侧刻度会落在绘图区外，被画到轴线下方。
  if (yTickValues.length >= 2) {
    yMin = Math.min(yMin, yTickValues[0]);
    yMax = Math.max(yMax, yTickValues[yTickValues.length - 1]);
  }
  const scaleY = (value) => plot.y1 - ((value - yMin) / (yMax - yMin)) * plotHeight;

  const yTicks = yTickValues.map((value) => ({ value, y: scaleY(value), label: formatValue(value) }));

  // x 刻度：标签太密时按步长抽稀。
  const maxLabels = Math.max(2, Math.floor(plotWidth / 74));
  const stride = Math.max(1, Math.ceil(shownCount / maxLabels));
  const shownCategories = shown.map((category, index) => ({
    key: category.key,
    label: category.label,
    x: positionAt(index),
    index: domain.from + index,
  }));
  const xTicks = shownCategories.filter((_, index) => index % stride === 0 || index === shownCategories.length - 1);

  // 系列几何
  const scenes = series.map((item, seriesIndex) => {
    const isHidden = hidden.has(seriesIndex);
    const valuesByCategory = new Map();
    item.x.forEach((xValue, pointIndex) => {
      valuesByCategory.set(String(xValue), item.y[pointIndex]);
    });
    const aligned = shownCategories.map((category) => {
      const raw = valuesByCategory.get(category.key);
      return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
    });
    const points = aligned.map((value, index) => ({
      x: shownCategories[index].x,
      y: value === null ? null : scaleY(value),
      value,
      category: shownCategories[index].label,
      categoryIndex: shownCategories[index].index,
    }));
    return {
      name: item.name,
      color: PALETTE[seriesIndex % PALETTE.length],
      hidden: isHidden,
      values: aligned,
      points,
    };
  });

  // 柱状几何：同一类别内并排
  if (kind === "bar") {
    const visibleIndexes = scenes.map((item, index) => (item.hidden ? -1 : index)).filter((index) => index >= 0);
    const slot = plotWidth / Math.max(shownCount, 1);
    const groupWidth = slot * 0.72;
    const barWidth = groupWidth / Math.max(visibleIndexes.length, 1);
    scenes.forEach((item, seriesIndex) => {
      if (item.hidden) {
        return;
      }
      const order = visibleIndexes.indexOf(seriesIndex);
      item.bars = item.points.map((point) => {
        if (point.value === null) {
          return null;
        }
        const x = point.x - groupWidth / 2 + order * barWidth;
        const y = scaleY(point.value);
        return { x, y, width: Math.max(barWidth - 1, 1), height: Math.max(plot.y1 - y, 0), value: point.value };
      });
    });
  }

  // 图例：底部一行，超出宽度则截断
  const legend = [];
  if (showLegend) {
    let cursor = plot.x0;
    series.forEach((item, index) => {
      const label = item.name;
      const itemWidth = 20 + label.length * 7.2 + 16;
      legend.push({
        name: label,
        color: PALETTE[index % PALETTE.length],
        hidden: hidden.has(index),
        index,
        x: cursor,
        y: height - 14,
        width: itemWidth,
      });
      cursor += itemWidth;
    });
  }

  return {
    width,
    height,
    kind,
    theme,
    plot,
    title: spec.title || "",
    xlabel: spec.xlabel || "",
    ylabel: spec.ylabel || "",
    yTicks,
    xTicks,
    categories: shownCategories,
    series: scenes,
    legend,
    domain: { ...domain, total },
    showLegend,
  };
}

function normalizeDomain(domain, total) {
  if (!domain || total === 0) {
    return { from: 0, to: Math.max(total - 1, 0) };
  }
  let from = Math.max(0, Math.min(domain.from, total - 1));
  let to = Math.max(from, Math.min(domain.to, total - 1));
  if (to - from < 1) {
    to = Math.min(total - 1, from + 1);
    from = Math.max(0, to - 1);
  }
  return { from, to };
}

/** 场景里离某个像素 x 最近的数据点（用于 tooltip）。 */
export function nearestPoint(scene, pixelX) {
  const candidates = [];
  scene.categories.forEach((category, index) => {
    candidates.push({ index, x: category.x, label: category.label });
  });
  if (!candidates.length) {
    return null;
  }
  let best = candidates[0];
  candidates.forEach((candidate) => {
    if (Math.abs(candidate.x - pixelX) < Math.abs(best.x - pixelX)) {
      best = candidate;
    }
  });
  return best;
}

/* ==========================================================================
   纯函数：SVG 输出
   ========================================================================== */

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** 场景 → SVG 标记字符串。 */
export function renderSvg(scene) {
  const { width, height, plot, theme } = scene;
  const parts = [];
  const push = (chunk) => parts.push(chunk);

  push(
    '<svg class="data-figure__svg" xmlns="http://www.w3.org/2000/svg" ' +
      'viewBox="0 0 ' + width + " " + height + '" width="100%" height="' + height + '" ' +
      'role="img" style="display:block;overflow:visible">'
  );

  if (scene.title) {
    push(
      '<text x="' + plot.x0 + '" y="20" fill="' + theme.ink + '" font-size="15" font-weight="600">' +
        esc(scene.title) + "</text>"
    );
  }

  // 网格
  scene.yTicks.forEach((tick) => {
    push(
      '<line x1="' + plot.x0 + '" y1="' + tick.y.toFixed(1) + '" x2="' + plot.x1 + '" y2="' +
        tick.y.toFixed(1) + '" stroke="' + theme.grid + '" stroke-width="1"/>'
    );
  });

  // 坐标轴
  push(
    '<line x1="' + plot.x0 + '" y1="' + plot.y0 + '" x2="' + plot.x0 + '" y2="' + plot.y1 +
      '" stroke="' + theme.axis + '" stroke-width="1"/>'
  );
  push(
    '<line x1="' + plot.x0 + '" y1="' + plot.y1 + '" x2="' + plot.x1 + '" y2="' + plot.y1 +
      '" stroke="' + theme.axis + '" stroke-width="1"/>'
  );

  // y 刻度标签
  scene.yTicks.forEach((tick) => {
    push(
      '<text x="' + (plot.x0 - 8) + '" y="' + (tick.y + 4).toFixed(1) +
        '" fill="' + theme.muted + '" font-size="11" text-anchor="end">' + esc(tick.label) + "</text>"
    );
  });

  // x 刻度标签
  scene.xTicks.forEach((tick) => {
    push(
      '<text x="' + tick.x.toFixed(1) + '" y="' + (plot.y1 + 16) +
        '" fill="' + theme.muted + '" font-size="11" text-anchor="middle">' + esc(tick.label) + "</text>"
    );
  });

  // 轴标题
  if (scene.xlabel) {
    push(
      '<text x="' + ((plot.x0 + plot.x1) / 2).toFixed(1) + '" y="' + (plot.y1 + 34) +
        '" fill="' + theme.muted + '" font-size="12" text-anchor="middle">' + esc(scene.xlabel) + "</text>"
    );
  }
  if (scene.ylabel) {
    const centerY = (plot.y0 + plot.y1) / 2;
    push(
      '<text x="14" y="' + centerY.toFixed(1) + '" fill="' + theme.muted + '" font-size="12" ' +
        'text-anchor="middle" transform="rotate(-90 14 ' + centerY.toFixed(1) + ')">' +
        esc(scene.ylabel) + "</text>"
    );
  }

  // 系列
  scene.series.forEach((item) => {
    if (item.hidden) {
      return;
    }
    if (scene.kind === "bar") {
      (item.bars || []).forEach((bar) => {
        if (!bar) {
          return;
        }
        push(
          '<rect x="' + bar.x.toFixed(1) + '" y="' + bar.y.toFixed(1) + '" width="' +
            bar.width.toFixed(1) + '" height="' + bar.height.toFixed(1) + '" fill="' + item.color +
            '" rx="1"><title>' + esc(item.name + "：" + formatValue(bar.value)) +
            "</title></rect>"
        );
      });
      return;
    }

    const segments = [];
    let current = [];
    item.points.forEach((point) => {
      if (point.y === null) {
        if (current.length) {
          segments.push(current);
          current = [];
        }
        return;
      }
      current.push(point);
    });
    if (current.length) {
      segments.push(current);
    }

    segments.forEach((segment) => {
      const coordinates = segment.map((point) => point.x.toFixed(1) + "," + point.y.toFixed(1)).join(" ");
      if (scene.kind === "area") {
        const first = segment[0];
        const last = segment[segment.length - 1];
        push(
          '<polygon points="' + first.x.toFixed(1) + "," + plot.y1 + " " + coordinates + " " +
            last.x.toFixed(1) + "," + plot.y1 + '" fill="' + item.color +
            '" fill-opacity="0.18" stroke="none"/>'
        );
      }
      push(
        '<polyline points="' + coordinates + '" fill="none" stroke="' + item.color +
          '" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
      );
      if (scene.kind === "scatter") {
        segment.forEach((point) => {
          push(
            '<circle cx="' + point.x.toFixed(1) + '" cy="' + point.y.toFixed(1) +
              '" r="3.5" fill="' + item.color + '"/>'
          );
        });
      }
    });
  });

  // 图例（可点击：data-legend-index）
  if (scene.showLegend) {
    scene.legend.forEach((item) => {
      const opacity = item.hidden ? 0.35 : 1;
      push(
        '<g class="data-figure__legend-item" data-legend-index="' + item.index +
          '" style="cursor:pointer" opacity="' + opacity + '">' +
          '<rect x="' + item.x.toFixed(1) + '" y="' + (item.y - 9) + '" width="' + item.width +
          '" height="20" fill="transparent"/>' +
          '<line x1="' + item.x.toFixed(1) + '" y1="' + item.y + '" x2="' + (item.x + 14) +
          '" y2="' + item.y + '" stroke="' + item.color + '" stroke-width="3"/>' +
          '<text x="' + (item.x + 19).toFixed(1) + '" y="' + (item.y + 4) +
          '" fill="' + scene.theme.ink + '" font-size="12">' + esc(item.name) + "</text>" +
          "</g>"
      );
    });
  }

  // 交互层：命中区、竖向参考线、框选矩形
  push(
    '<rect class="data-figure__hit" x="' + plot.x0 + '" y="' + plot.y0 + '" width="' +
      (plot.x1 - plot.x0) + '" height="' + (plot.y1 - plot.y0) + '" fill="transparent"/>'
  );
  push(
    '<line class="data-figure__guide" x1="0" y1="' + plot.y0 + '" x2="0" y2="' + plot.y1 +
      '" stroke="' + theme.muted + '" stroke-width="1" stroke-dasharray="3 3" style="display:none"/>'
  );
  push(
    '<rect class="data-figure__brush" x="0" y="' + plot.y0 + '" width="0" height="' +
      (plot.y1 - plot.y0) + '" fill="' + theme.ink + '" fill-opacity="0.08" style="display:none"/>'
  );

  push("</svg>");
  return parts.join("");
}

/* ==========================================================================
   纯函数：参数量轴（timeline）
   ========================================================================== */

function pickColumn(columns, preferred, patterns, what) {
  if (preferred) {
    if (!columns.includes(preferred)) {
      throw new Error("找不到" + what + "列「" + preferred + "」；可用列：" + columns.join("、"));
    }
    return preferred;
  }
  for (const pattern of patterns) {
    const hit = columns.find((column) => pattern.test(column));
    if (hit) {
      return hit;
    }
  }
  return "";
}

/**
 * 行数据 → 模型列表（参数量单位：十亿）。
 * 列名可用 include 参数指定，也可以通过常见列名自动识别。
 */
export function rowsToModels(rows, spec = {}) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return { models: [], deployment: [], columns: {}, extras: [] };
  }
  const columns = Object.keys(rows[0] || {});
  const nameColumn = pickColumn(columns, spec.label, [/^model$/i, /^name$/i, /model/], "Model Name");
  const totalColumn = pickColumn(columns, spec.x, [/total.*param/i, /^params?$/i, /^size$/i, /param/], "Parameters");
  const activeColumn = pickColumn(columns, (spec.y && spec.y[0]) || "", [/active.*param/i, /active_param/], "Active Parameters");
  const deploymentColumn = pickColumn(columns, spec.series, [/deployment/i, /category/i, /domain/i, /deployment/], "Deployment");

  if (!totalColumn) {
    throw new Error("找不到参数量列；可用列：" + columns.join("、"));
  }

  const used = new Set([nameColumn, totalColumn, activeColumn, deploymentColumn].filter(Boolean));
  const extras = columns.filter((column) => !used.has(column));

  const models = rows
    .map((row, index) => {
      const total = typeof row[totalColumn] === "number" ? row[totalColumn] : Number.NaN;
      return {
        index,
        name: nameColumn ? String(row[nameColumn] === null ? "未命名" : row[nameColumn]) : "模型 " + (index + 1),
        total,
        active: activeColumn && typeof row[activeColumn] === "number" ? row[activeColumn] : null,
        deployment: deploymentColumn && row[deploymentColumn] !== null ? String(row[deploymentColumn]) : "未分组",
        extras: extras.map((column) => ({ label: column, value: row[column] })),
      };
    })
    .filter((model) => Number.isFinite(model.total) && model.total > 0);

  if (!models.length) {
    throw new Error("参数量列「" + totalColumn + "」里没有有效的正数");
  }

  const deployment = [];
  models.forEach((model) => {
    if (!deployment.includes(model.deployment)) {
      deployment.push(model.deployment);
    }
  });

  return {
    models,
    deployment,
    columns: { name: nameColumn, total: totalColumn, active: activeColumn, deployment: deploymentColumn },
    extras,
  };
}

/** 参数量（十亿）→ 1B / 10B / 100B / 1T 这样的短标签。 */
export function formatParams(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (value >= 1000) {
    return formatValue(value / 1000) + "T";
  }
  if (value >= 1) {
    return formatValue(value) + "B";
  }
  return formatValue(value * 1000) + "M";
}

function charWidth(char) {
  // 粗略估计：CJK 字形宽度约为 ASCII 的两倍
  return char.charCodeAt(0) > 0x2e80 ? 10 : 5.6;
}

function estimateWidth(text) {
  let total = 0;
  for (const char of String(text === null || text === undefined ? "" : text)) {
    total += charWidth(char);
  }
  return total;
}

function truncateToWidth(text, maxWidth) {
  const value = String(text === null || text === undefined ? "" : text);
  if (estimateWidth(value) <= maxWidth) {
    return value;
  }
  let out = "";
  let width = 0;
  for (const char of value) {
    const next = charWidth(char);
    if (width + next > maxWidth - 8) {
      break;
    }
    out += char;
    width += next;
  }
  return out + "…";
}

/**
 * 模型列表 → 参数轴场景几何。
 * x 轴为对数刻度的参数量；柱高统一，MoE 用同色相深浅按 激活/总 比例分割柱身；
 * 轴下方按应用场景画横向分组条。
 */
/**
 * 决定时间轴 SVG 的画布宽度：容器放不下时按需加宽，超出部分由容器横向滚动。
 * 目标是最小柱间距不小于 30px（柱宽约为其 0.55 倍），避免模型一多就糊成一片。
 */
export function timelineWidth(totals, logMin, span, margin, containerWidth, count) {
  const usable = (value) => Math.max(value - margin.left - margin.right, 40);
  // 底线：每个模型至少分到 44px，且容器本身要铺满
  let width = Math.max(
    containerWidth,
    Math.round(count * 44 + margin.left + margin.right),
    Math.round(span * 180 + margin.left + margin.right)
  );

  for (let attempt = 0; attempt < 4; attempt += 1) {
    const plotW = usable(width);
    const positions = totals
      .map((value) => ((Math.log10(value) - logMin) / span) * plotW)
      .sort((a, b) => a - b);
    let minGap = Infinity;
    for (let i = 1; i < positions.length; i += 1) {
      minGap = Math.min(minGap, positions[i] - positions[i - 1]);
    }
    if (!Number.isFinite(minGap) || minGap >= 30) {
      break;
    }
    // 一次最多放大 2.5 倍，最多试 4 轮：应对「绝大多数模型挤在同一数量级」的情况
    width = Math.min(9000, Math.ceil(width * Math.min(2.5, 30 / Math.max(minGap, 1))));
  }
  return width;
}

export function buildTimelineScene(data, spec = {}, options = {}) {
  const containerWidth = Math.max(320, Math.round(options.width || 860));
  const theme = options.theme || THEMES.light;
  const highlight = Number.isInteger(options.highlight) ? options.highlight : -1;
  const models = data.models || [];
  const deployment = data.deployment || [];

  const totals = models.map((model) => model.total);
  const minTotal = Math.min(...totals);
  const maxTotal = Math.max(...totals);
  const logMin = Math.floor(Math.log10(minTotal));
  const logMax = Math.max(Math.ceil(Math.log10(maxTotal)), logMin + 1);
  const span = logMax - logMin;

  const colorOf = new Map(deployment.map((name, index) => [name, PALETTE[index % PALETTE.length]]));

  const margin = { left: 62, right: 26, top: spec.title ? 50 : 16 };
  // 竖排模型名是向上生长的，标签带高度按最长名字算，避免和标题打架
  const longestWidth = models.reduce((max, model) => Math.max(max, estimateWidth(model.name)), 0);
  const labelBand = Math.min(170, Math.max(64, Math.round(longestWidth + 14)));
  const specHeight = Math.max(300, Math.round(spec.height || 460));
  const barHeight = Math.max(96, Math.min(170, Math.round(specHeight * 0.3)));
  const width = timelineWidth(totals, logMin, span, margin, containerWidth, models.length);
  const plot = {
    x0: margin.left,
    x1: width - margin.right,
    y0: margin.top + labelBand,
    y1: margin.top + labelBand + barHeight,
  };
  const plotWidth = Math.max(plot.x1 - plot.x0, 40);
  const xOf = (value) => plot.x0 + ((Math.log10(value) - logMin) / span) * plotWidth;

  // 刻度：每个数量级取 1/2/5，1 为主刻度（带标签与网格线）
  const ticks = [];
  for (let decade = logMin; decade <= logMax; decade += 1) {
    for (const multiplier of [1, 2, 5]) {
      const value = multiplier * Math.pow(10, decade);
      if (value < Math.pow(10, logMin) - 1e-9 || value > Math.pow(10, logMax) + 1e-9) {
        continue;
      }
      ticks.push({
        value,
        x: xOf(value),
        major: multiplier === 1,
        label: formatParams(value),
      });
    }
  }

  // 柱宽按最小间距自适应，避免密集处糊成一片
  const sorted = models.slice().sort((a, b) => xOf(a.total) - xOf(b.total));
  let minGap = Infinity;
  for (let i = 1; i < sorted.length; i += 1) {
    minGap = Math.min(minGap, xOf(sorted[i].total) - xOf(sorted[i - 1].total));
  }
  const barWidth = sorted.length > 1 ? Math.max(7, Math.min(26, minGap * 0.55)) : 22;

  const geometry = sorted.map((model) => {
    const ratio = model.active && model.active > 0 && model.active < model.total
      ? model.active / model.total
      : null;
    const isHot = model.index === highlight;
    const cx = xOf(model.total);
    const barW = barWidth + (isHot ? 6 : 0);
    const top = plot.y0 - (isHot ? 9 : 0);
    const totalHeight = plot.y1 - top;
    return {
      index: model.index,
      name: model.name,
      deployment: model.deployment,
      total: model.total,
      active: model.active,
      ratio,
      cx,
      color: colorOf.get(model.deployment) || PALETTE[0],
      highlight: isHot,
      extras: model.extras,
      bar: { x: cx - barW / 2, y: top, width: barW, height: totalHeight },
      activeBar: ratio === null
        ? null
        : { x: cx - barW / 2, y: plot.y1 - totalHeight * ratio, width: barW, height: totalHeight * ratio },
    };
  });

  // 场景分组条：横跨该场景所有模型
  const bandHeight = 19;
  const bandGap = 7;
  const bandsTop = plot.y1 + 46;
  const bands = deployment.map((name, index) => {
    const members = geometry.filter((model) => model.deployment === name);
    const xs = members.map((model) => model.cx);
    let x0 = Math.min(...xs) - barWidth / 2 - 6;
    let x1 = Math.max(...xs) + barWidth / 2 + 6;
    if (x1 - x0 < 52) {
      const middle = (x0 + x1) / 2;
      x0 = middle - 26;
      x1 = middle + 26;
    }
    return {
      name,
      x0: Math.max(x0, 2),
      x1: Math.min(x1, width - 2),
      y: bandsTop + index * (bandHeight + bandGap),
      height: bandHeight,
      color: colorOf.get(name) || PALETTE[0],
      count: members.length,
    };
  });

  const contentBottom = bands.length ? bands[bands.length - 1].y + bandHeight : plot.y1 + 24;
  const height = Math.max(specHeight, Math.round(contentBottom + 14));

  return {
    kind: "timeline",
    width,
    height,
    theme,
    plot,
    title: spec.title || "",
    xlabel: spec.xlabel || "",
    ticks,
    models: geometry,
    bands,
    labelWidth: Math.max(28, labelBand - 10),
    highlight,
    containerWidth,
    scrollable: width > containerWidth + 1,
  };
}

/** 命中测试：鼠标落在哪根柱子上（返回模型 index，未命中为 -1）。 */
export function timelineHitTest(scene, x, y) {
  if (!scene || scene.kind !== "timeline") {
    return -1;
  }
  if (y < scene.plot.y0 - 12 || y > scene.plot.y1 + 6) {
    return -1;
  }
  let best = -1;
  let bestDistance = Infinity;
  scene.models.forEach((model) => {
    const distance = Math.abs(model.cx - x);
    if (distance <= model.bar.width / 2 + 6 && distance < bestDistance) {
      best = model.index;
      bestDistance = distance;
    }
  });
  return best;
}

/** 参数轴场景 → SVG 标记字符串。 */
export function renderTimelineSvg(scene) {
  const { width, height, plot, theme } = scene;
  const parts = [];
  const push = (chunk) => parts.push(chunk);

  push(
    '<svg class="data-figure__svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + width + " " +
      height + '" width="' + width + '" height="' + height + '" role="img" ' +
      'style="display:block;overflow:visible">'
  );

  if (scene.title) {
    push(
      '<text x="' + plot.x0 + '" y="20" fill="' + theme.ink + '" font-size="15" font-weight="600">' +
        esc(scene.title) + "</text>"
    );
  }

  // 主刻度的竖向网格
  scene.ticks
    .filter((tick) => tick.major)
    .forEach((tick) => {
      push(
        '<line x1="' + tick.x.toFixed(1) + '" y1="' + plot.y0 + '" x2="' + tick.x.toFixed(1) +
          '" y2="' + plot.y1 + '" stroke="' + theme.grid + '" stroke-width="1"/>'
      );
    });

  // 选中柱的放大光晕（画在最底层）
  scene.models
    .filter((model) => model.highlight)
    .forEach((model) => {
      push(
        '<rect x="' + (model.bar.x - 4).toFixed(1) + '" y="' + (model.bar.y - 4).toFixed(1) +
          '" width="' + (model.bar.width + 8).toFixed(1) + '" height="' + (model.bar.height + 7).toFixed(1) +
          '" rx="6" fill="' + model.color + '" fill-opacity="0.25"/>'
      );
    });

  // 柱子：密集模型实心，MoE 用浅色总参 + 深色激活
  scene.models.forEach((model) => {
    const bar = model.bar;
    if (model.activeBar) {
      push(
        '<rect x="' + bar.x.toFixed(1) + '" y="' + bar.y.toFixed(1) + '" width="' + bar.width.toFixed(1) +
          '" height="' + bar.height.toFixed(1) + '" rx="2" fill="' + model.color + '" fill-opacity="0.3"/>'
      );
      push(
        '<rect x="' + model.activeBar.x.toFixed(1) + '" y="' + model.activeBar.y.toFixed(1) +
          '" width="' + model.activeBar.width.toFixed(1) + '" height="' + model.activeBar.height.toFixed(1) +
          '" rx="2" fill="' + model.color + '"/>'
      );
      return;
    }
    push(
      '<rect x="' + bar.x.toFixed(1) + '" y="' + bar.y.toFixed(1) + '" width="' + bar.width.toFixed(1) +
        '" height="' + bar.height.toFixed(1) + '" rx="2" fill="' + model.color + '" fill-opacity="0.88"/>'
    );
  });

  // 竖排模型名（放在柱顶上方，超宽按可用宽度截断）
  scene.models.forEach((model) => {
    const label = truncateToWidth(model.name, scene.labelWidth);
    const anchorX = (model.cx + 3.6).toFixed(1);
    const anchorY = (model.bar.y - 7).toFixed(1);
    push(
      '<text x="' + anchorX + '" y="' + anchorY + '" fill="' + theme.muted + '" font-size="10" ' +
        'text-anchor="start" transform="rotate(-90 ' + anchorX + " " + anchorY + ')">' +
        esc(label) + "</text>"
    );
  });

  // 轴与刻度
  push(
    '<line x1="' + plot.x0 + '" y1="' + plot.y1 + '" x2="' + plot.x1 + '" y2="' + plot.y1 +
      '" stroke="' + theme.axis + '" stroke-width="1"/>'
  );
  scene.ticks.forEach((tick) => {
    if (tick.major) {
      push(
        '<text x="' + tick.x.toFixed(1) + '" y="' + (plot.y1 + 15) +
          '" fill="' + theme.muted + '" font-size="11" text-anchor="middle">' + esc(tick.label) + "</text>"
      );
    } else {
      push(
        '<line x1="' + tick.x.toFixed(1) + '" y1="' + plot.y1 + '" x2="' + tick.x.toFixed(1) +
          '" y2="' + (plot.y1 + 4) + '" stroke="' + theme.axis + '" stroke-width="1"/>'
      );
    }
  });

  if (scene.xlabel) {
    push(
      '<text x="' + ((plot.x0 + plot.x1) / 2).toFixed(1) + '" y="' + (plot.y1 + 33) +
        '" fill="' + theme.muted + '" font-size="12" text-anchor="middle">' + esc(scene.xlabel) + "</text>"
    );
  }

  // 场景分组条
  scene.bands.forEach((band) => {
    push(
      '<rect x="' + band.x0.toFixed(1) + '" y="' + band.y.toFixed(1) + '" width="' +
        (band.x1 - band.x0).toFixed(1) + '" height="' + band.height +
        '" rx="9" fill="' + band.color + '" fill-opacity="0.16" stroke="' + band.color +
        '" stroke-opacity="0.55" stroke-width="1"/>'
    );
    push(
      '<text x="' + (band.x0 + 9).toFixed(1) + '" y="' + (band.y + 13.5) +
        '" fill="' + theme.ink + '" font-size="11">' + esc(band.name) +
        "（" + band.count + "）</text>"
    );
  });

  push(
    '<rect class="data-figure__hit" x="' + plot.x0 + '" y="' + (plot.y0 - 12) + '" width="' +
      (plot.x1 - plot.x0) + '" height="' + (plot.y1 - plot.y0 + 18) + '" fill="transparent"/>'
  );

  push("</svg>");
  return parts.join("");
}

/* ==========================================================================
   DOM 层
   ========================================================================== */

const figures = [];
let themeObserver = null;

function storedTheme() {
  try {
    const value = window.localStorage.getItem("theme");
    return value === "dark" || value === "light" ? value : null;
  } catch (error) {
    return null;
  }
}

export function currentThemeKind() {
  const attribute = document.documentElement.getAttribute("data-theme");
  if (attribute === "dark") {
    return "dark";
  }
  const stored = storedTheme();
  if (stored) {
    return stored;
  }
  if (attribute === "light") {
    return "light";
  }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export async function fetchDataset(src) {
  const response = await fetch(src, {
    headers: { Accept: "text/csv, application/json;q=0.9, */*;q=0.8" },
  });
  if (!response.ok) {
    const error = new Error("数据加载失败（HTTP " + response.status + "）");
    error.status = response.status;
    throw error;
  }
  const text = await response.text();
  return /\.json(?:$|[?#])/i.test(src) ? parseJson(text) : parseCsv(text);
}

function readSpec(element) {
  const data = element.dataset;
  const bool = (value) => String(value) === "true";
  const number = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  return {
    id: element.id,
    src: data.src || "",
    type: data.type || "line",
    x: data.x || "",
    y: String(data.y || "").split(",").map((item) => item.trim()).filter(Boolean),
    series: data.series || "",
    sort: data.sort || "file",
    label: data.label || "",
    title: data.title || "",
    xlabel: data.xlabel || "",
    ylabel: data.ylabel || "",
    height: number(data.height, 420),
    showLegend: data.showLegend !== "false",
  };
}

function setStatus(element, message, options = {}) {
  const state = options.state || "loading";
  element.dataset.state = state;
  element.setAttribute("aria-busy", state === "loading" ? "true" : "false");

  let status = element.querySelector("." + STATUS_CLASS);
  if (!status) {
    status = document.createElement("p");
    status.className = STATUS_CLASS;
    element.appendChild(status);
  }
  status.textContent = message;
  if (options.href) {
    const link = document.createElement("a");
    link.href = options.href;
    link.textContent = "查看原始数据";
    link.style.marginLeft = "0.6em";
    status.appendChild(link);
  }
}

function clearStatus(element) {
  const status = element.querySelector("." + STATUS_CLASS);
  if (status) {
    status.remove();
  }
}

function draw(record) {
  const { element, spec } = record;
  const width =
    element.clientWidth ||
    (element.parentElement ? element.parentElement.clientWidth : 0) ||
    720;
  const theme = THEMES[currentThemeKind()] || THEMES.light;

  if (spec.type === "timeline") {
    record.scene = buildTimelineScene(record.modelData, spec, {
      width,
      theme,
      highlight: record.highlight === undefined ? -1 : record.highlight,
    });
    element.innerHTML = renderTimelineSvg(record.scene);
    element.dataset.scroll = record.scene.scrollable ? "true" : "false";
  } else {
    record.scene = buildScene(record.series, spec, {
      width,
      height: spec.height,
      theme,
      hidden: record.hidden,
      domain: record.domain,
    });
    element.innerHTML = renderSvg(record.scene);
    element.dataset.scroll = "false";
  }

  element.dataset.state = "ready";
  element.setAttribute("aria-busy", "false");
  element.setAttribute("role", "img");
  if (!element.getAttribute("aria-label")) {
    element.setAttribute("aria-label", spec.title || "交互式图表：" + spec.src);
  }
  // innerHTML 会清空子节点，把 tooltip 节点接回去，避免悬停时反复重建。
  if (record.tooltipNode) {
    element.appendChild(record.tooltipNode);
  }
  // 事件统一在 setup() 里挂一次：换成渲染后 SVG 会重建，挂在容器上才不会丢。
}

function tooltipOf(record) {
  let tooltip = record.element.querySelector("." + TOOLTIP_CLASS);
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = TOOLTIP_CLASS;
    tooltip.hidden = true;
    record.element.appendChild(tooltip);
  }
  record.tooltipNode = tooltip;
  return tooltip;
}

function tooltipTitle(tooltip, text) {
  const head = document.createElement("div");
  head.className = "data-figure__tooltip-title";
  head.textContent = text;
  tooltip.appendChild(head);
}

function tooltipRow(tooltip, options) {
  const line = document.createElement("div");
  line.className = "data-figure__tooltip-row" + (options.meta ? " data-figure__tooltip-meta" : "");
  if (options.color) {
    const dot = document.createElement("span");
    dot.className = "data-figure__tooltip-dot";
    dot.style.background = options.color;
    line.appendChild(dot);
  }
  const text = document.createElement("span");
  text.textContent = options.text;
  line.appendChild(text);
  tooltip.appendChild(line);
  return line;
}

/** 事件坐标 → 场景坐标（时间轴是 1:1 像素，需要叠加横向滚动量）。 */
function localPoint(record, event) {
  const rect = record.element.getBoundingClientRect();
  const scene = record.scene;
  if (scene.kind === "timeline") {
    return {
      x: event.clientX - (rect.left || 0) + (record.element.scrollLeft || 0),
      y: event.clientY - (rect.top || 0),
      scale: 1,
    };
  }
  const width = rect.width || scene.width;
  const scale = scene.width / width;
  return {
    x: (event.clientX - (rect.left || 0)) * scale,
    y: (event.clientY - (rect.top || 0)) * scale,
    scale,
  };
}

function hideTooltip(record) {
  const tooltip = tooltipOf(record);
  tooltip.hidden = true;
  const guide = record.element.querySelector(".data-figure__guide");
  if (guide) {
    guide.style.display = "none";
  }
}

function attachSeriesHover(record) {
  const { element } = record;

  element.addEventListener("mousemove", (event) => {
    const scene = record.scene;
    if (!scene || scene.kind === "timeline") {
      return;
    }
    const point = localPoint(record, event);
    if (point.x < scene.plot.x0 || point.x > scene.plot.x1) {
      hideTooltip(record);
      return;
    }
    const nearest = nearestPoint(scene, point.x);
    if (!nearest) {
      hideTooltip(record);
      return;
    }
    const guide = element.querySelector(".data-figure__guide");
    if (guide) {
      guide.style.display = "";
      guide.setAttribute("x1", nearest.x);
      guide.setAttribute("x2", nearest.x);
    }

    const tooltip = tooltipOf(record);
    tooltip.textContent = "";
    tooltipTitle(tooltip, nearest.label);
    scene.series
      .filter((item) => !item.hidden)
      .forEach((item) => {
        tooltipRow(tooltip, {
          color: item.color,
          text: item.name + "：" + formatValue(item.values[nearest.index]),
        });
      });

    tooltip.hidden = false;
    const containerWidth = element.clientWidth || scene.width;
    const left = (nearest.x / scene.width) * containerWidth;
    tooltip.style.left = Math.min(Math.max(left + 12, 8), Math.max(containerWidth - 180, 8)) + "px";
    tooltip.style.top = scene.plot.y0 / (point.scale || 1) + 4 + "px";
  });

  element.addEventListener("mouseleave", () => hideTooltip(record));
}

/** 参数轴图表：悬停放大柱子，并在柱顶上方浮出详情。 */
function showTimelineTooltip(record, index) {
  const scene = record.scene;
  const model = scene.models.find((item) => item.index === index);
  if (!model) {
    return;
  }
  const tooltip = tooltipOf(record);

  tooltip.textContent = "";
  tooltipTitle(tooltip, model.name);
  tooltipRow(tooltip, { color: model.color, text: "Deployment：" + model.deployment, meta: true });
  tooltipRow(tooltip, { text: "Total parameters：" + formatParams(model.total) });
  if (model.ratio !== null) {
    const percent = (model.ratio * 100).toFixed(model.ratio < 0.1 ? 1 : 0);
    tooltipRow(tooltip, {
      text: "Activated parameters：" + formatParams(model.active) + "（" + percent + "%）",
    });
  }
  model.extras.forEach((extra) => {
    if (extra.value === null || extra.value === undefined || extra.value === "") {
      return;
    }
    tooltipRow(tooltip, { text: extra.label + "：" + formatValue(extra.value), meta: true });
  });

  tooltip.hidden = false;
  // 时间轴可能横向滚动：先换算成可见区坐标，夹紧后再换回内容坐标，
  // 这样详情框不会撑大滚动宽度，也不会被容器裁掉。
  const element = record.element;
  const scrollLeft = element.scrollLeft || 0;
  const visibleWidth = element.clientWidth || scene.containerWidth || scene.width;
  const tooltipWidth = tooltip.offsetWidth || 200;
  const desired = model.cx + 10 - scrollLeft;
  const leftVisible = Math.min(Math.max(desired, 6), Math.max(visibleWidth - tooltipWidth - 6, 6));
  tooltip.style.left = (leftVisible + scrollLeft) + "px";
  const height = tooltip.offsetHeight || 96;
  tooltip.style.top = Math.max(0, model.bar.y - height - 10) + "px";
}

function attachTimelineHover(record) {
  const { element } = record;

  element.addEventListener("mousemove", (event) => {
    const scene = record.scene;
    if (!scene || scene.kind !== "timeline") {
      return;
    }
    const point = localPoint(record, event);
    const index = timelineHitTest(scene, point.x, point.y);
    if (index === record.highlight) {
      return;
    }
    record.highlight = index;
    if (index === -1) {
      hideTooltip(record);
      draw(record);
      return;
    }
    draw(record);
    showTimelineTooltip(record, index);
  });

  // 滚动时旧位置会失效，先把详情框收起来，等下一次移动再显示
  element.addEventListener("scroll", () => {
    hideTooltip(record);
  }, { passive: true });

  // 桌面端鼠标滚轮 → 横向滚动：滚到尽头再交还给页面，避免把页面滚动卡死
  element.addEventListener("wheel", (event) => {
    const scene = record.scene;
    if (!scene || !scene.scrollable) {
      return;
    }
    if (Math.abs(event.deltaX || 0) > Math.abs(event.deltaY || 0)) {
      return;
    }
    const maxScroll = (element.scrollWidth || 0) - (element.clientWidth || 0);
    if (maxScroll <= 0) {
      return;
    }
    const delta = event.deltaY || 0;
    const canScrollRight = element.scrollLeft < maxScroll - 1;
    const canScrollLeft = element.scrollLeft > 0;
    if ((delta > 0 && canScrollRight) || (delta < 0 && canScrollLeft)) {
      element.scrollLeft = Math.max(0, Math.min(maxScroll, element.scrollLeft + delta));
      if (typeof event.preventDefault === "function") {
        event.preventDefault();
      }
    }
  }, { passive: false });

  element.addEventListener("mouseleave", () => {
    if (record.highlight !== -1) {
      record.highlight = -1;
      hideTooltip(record);
      draw(record);
    }
  });
}

function attachInteractions(record) {
  const { element } = record;
  element.addEventListener("click", (event) => {
    const target = event.target;
    const item = target && target.closest ? target.closest("[data-legend-index]") : null;
    if (!item) {
      return;
    }
    const index = Number(item.getAttribute("data-legend-index"));
    if (record.hidden.has(index)) {
      record.hidden.delete(index);
    } else {
      record.hidden.add(index);
    }
    draw(record);
  });

  // 拖拽框选缩放 + 双击复位
  let dragStart = null;
  const brush = () => element.querySelector(".data-figure__brush");

  element.addEventListener("mousedown", (event) => {
    const scene = record.scene;
    if (!scene || scene.kind === "timeline") {
      return;
    }
    const svg = element.querySelector("svg");
    if (!svg || !svg.contains(event.target)) {
      return;
    }
    const rect = svg.getBoundingClientRect();
    const scale = scene.width / (rect.width || scene.width);
    const x = (event.clientX - rect.left) * scale;
    if (x < scene.plot.x0 || x > scene.plot.x1) {
      return;
    }
    dragStart = x;
  });

  element.addEventListener("mousemove", (event) => {
    const scene = record.scene;
    if (dragStart === null || !scene) {
      return;
    }
    const svg = element.querySelector("svg");
    if (!svg) {
      return;
    }
    const rect = svg.getBoundingClientRect();
    const scale = scene.width / (rect.width || scene.width);
    const x = Math.min(Math.max((event.clientX - rect.left) * scale, scene.plot.x0), scene.plot.x1);
    const box = brush();
    if (!box) {
      return;
    }
    box.style.display = "";
    box.setAttribute("x", Math.min(dragStart, x));
    box.setAttribute("width", Math.abs(x - dragStart));
  });

  const finish = () => {
    const scene = record.scene;
    if (dragStart === null || !scene) {
      dragStart = null;
      return;
    }
    if (scene.kind === "timeline") {
      dragStart = null;
      return;
    }
    const box = brush();
    const startX = dragStart;
    dragStart = null;
    if (!box) {
      return;
    }
    const width = Number(box.getAttribute("width") || 0);
    box.style.display = "none";
    box.setAttribute("width", 0);
    if (width < 12) {
      return;
    }
    const fromX = Math.min(startX, startX + width);
    const toX = Math.max(startX, startX + width);
    const from = indexAtX(scene, fromX);
    const to = indexAtX(scene, toX);
    if (to - from < 1) {
      return;
    }
    record.domain = { from, to };
    draw(record);
  };

  element.addEventListener("mouseup", finish);
  element.addEventListener("mouseleave", () => {
    if (dragStart !== null) {
      finish();
    }
  });
  element.addEventListener("dblclick", () => {
    if (record.domain) {
      record.domain = null;
      draw(record);
    }
  });
}

function indexAtX(scene, pixelX) {
  const shown = scene.categories;
  if (!shown.length) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Infinity;
  shown.forEach((category, index) => {
    const distance = Math.abs(category.x - pixelX);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return shown[bestIndex].index;
}

function bindResize(record) {
  if (record.resizeBound || typeof ResizeObserver === "undefined") {
    return;
  }
  record.resizeBound = true;
  let frame = 0;
  record.observer = new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const width = record.element.clientWidth;
      if (width && Math.abs(width - (record.scene ? record.scene.width : 0)) > 2) {
        draw(record);
      }
    });
  });
  record.observer.observe(record.element);
}

async function setup(element) {
  const spec = readSpec(element);
  element.style.minHeight = spec.height + "px";

  if (!spec.src) {
    setStatus(element, "图表配置缺少数据文件地址（src）。", { state: "error" });
    return;
  }

  setStatus(element, "Loading data…");
  const record = {
    element,
    spec,
    series: null,
    modelData: null,
    scene: null,
    hidden: new Set(),
    domain: null,
    highlight: -1,
    resizeBound: false,
  };
  figures.push(record);

  try {
    const dataset = await fetchDataset(spec.src);
    if (!dataset.rows.length) {
      setStatus(element, "数据文件里没有数据行。", { state: "empty", href: spec.src });
      return;
    }
    if (spec.type === "timeline") {
      record.modelData = rowsToModels(dataset.rows, spec);
    } else {
      record.series = rowsToSeries(dataset.rows, spec);
      if (!record.series.length) {
        setStatus(element, "没有可绘制的系列。", { state: "empty", href: spec.src });
        return;
      }
    }
    clearStatus(element);
    draw(record);
    attachInteractions(record);
    if (spec.type === "timeline") {
      attachTimelineHover(record);
    } else {
      attachSeriesHover(record);
    }
    bindResize(record);
  } catch (error) {
    if (window.console) {
      console.error("[data-figure] " + spec.src, error);
    }
    setStatus(element, (error && error.message) || "Failed to render chart.", {
      state: "error",
      href: spec.src,
    });
  }
}

function redrawAll() {
  figures.forEach((record) => {
    if (record.series || record.modelData) {
      draw(record);
    }
  });
}

function observeTheme() {
  if (themeObserver) {
    return;
  }
  themeObserver = new MutationObserver(() => {
    redrawAll();
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  if (window.matchMedia) {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      if (!storedTheme()) {
        redrawAll();
      }
    };
    if (query.addEventListener) {
      query.addEventListener("change", handler);
    } else if (query.addListener) {
      query.addListener(handler);
    }
  }
}

export function init() {
  if (typeof document === "undefined" || window.__dataFigureReady) {
    return;
  }
  const elements = Array.from(document.querySelectorAll(".data-figure__chart[data-src]"));
  if (elements.length === 0) {
    return;
  }
  window.__dataFigureReady = true;
  elements.forEach((element) => {
    setup(element);
  });
  observeTheme();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
}
