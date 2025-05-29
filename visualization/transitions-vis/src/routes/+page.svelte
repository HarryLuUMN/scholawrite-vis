<script lang="ts">
  import * as d3 from "d3";
  import { onMount } from "svelte";

  interface DSVRowExtended extends d3.DSVRowString {
    project: string;
    timestamp: string;
    author: string;
    "before text": string;
    "after text": string;
    label: string;
    "high-level": string;
  }

  interface TaskRecord {
    project: number;
    timestamp: number;
    author: number;
    before_text: string;
    after_text: string;
    label: string;
    high_level: string;
    step?: number;
  }

  const config = {
    width: 800,
    height: 600,
    margin: { top: 100, right: 40, bottom: 160, left: 40 },
    nodeWidth: 15,
    nodePadding: 0.2,
    targetColor: "#555",
    maxLinkWidth: 60,
    minLabelGap: 15,
    minBandSize: 30,
    timeAxisHeight: 80,
  };

  // Full Data Cache
  let allRecords: TaskRecord[] = [];

  // Currently displayed data (filtered)
  let records: TaskRecord[] = [];
  let filteredRecords: TaskRecord[] = [];

  let svgElement: SVGSVGElement;

  // UI state
  let loading = true;
  let error: string | null = null;

  let showSelfTransitions = false;

  // Display Mode: Detailed Label or High Level Label
  let showHighLevel = false;

  // highlighting mode
  type HighlightMode = "normal" | "highLevelHighlight";
  let highlightMode: HighlightMode = "normal";

  let highlightSource: string | null = null;
  let highlightTarget: string | null = null;

  let brushStepRange: [number, number] | null = null;

  const projects = ["All", "Project 1", "Project 2", "Project 3", "Project 4", "Project 5"];
  let selectedProject = "All";

  // Project Correspondence Number Mapping
  const projectMap: Record<string, number | null> = {
    "All": null,
    "Project 1": 1,
    "Project 2": 2,
    "Project 3": 3,
    "Project 4": 4,
    "Project 5": 5,
  };

  const labelData = [
    { label: "Idea Generation", high_level: "PLANNING", color: "#4E79A7" },
    { label: "Idea Organization", high_level: "PLANNING", color: "#A0CBE8" },
    { label: "Section Planning", high_level: "PLANNING", color: "#F28E2B" },
    { label: "Text Production", high_level: "IMPLEMENTATION", color: "#FFBE7D" },
    { label: "Object Insertion", high_level: "IMPLEMENTATION", color: "#59A14F" },
    { label: "Cross-reference", high_level: "REVISION", color: "#8CD17D" },
    { label: "Citation Integration", high_level: "REVISION", color: "#B6992D" },
    { label: "Macro Insertion", high_level: "REVISION", color: "#F1CE63" },
    { label: "Fluency", high_level: "REVISION", color: "#499894" },
    { label: "Coherence", high_level: "REVISION", color: "#86BCB6" },
    { label: "Clarity", high_level: "REVISION", color: "#E15759" },
    { label: "Structural", high_level: "REVISION", color: "#FF9D9A" },
    { label: "Linguistic Style", high_level: "REVISION", color: "#79706E" },
    { label: "Scientific Accuracy", high_level: "REVISION", color: "#BAB0AC" },
    { label: "Visual Formatting", high_level: "REVISION", color: "#D37295" },
  ];

  const highLevels = ["PLANNING", "IMPLEMENTATION", "REVISION"];
  const highLevelColors = {
    PLANNING: "#4E79A7",
    IMPLEMENTATION: "#F28E2B",
    REVISION: "#59A14F",
  };

  let xScaleStep: d3.ScaleLinear<number, number>;
  let brush: d3.BrushBehavior<any>;

  async function loadAllData() {
    loading = true;
    error = null;
    try {
      const data = await d3.csv<DSVRowExtended>("/data/scholawrite_train.xls");
      allRecords = data
        .map((d) => ({
          project: Number(d.project) || 0,
          timestamp: Number(d.timestamp) || 0,
          author: Number(d.author) || 0,
          before_text: d["before text"] || "",
          after_text: d["after text"] || "",
          label: d.label || "",
          high_level: (d["high-level"] || "").toUpperCase(),
        }))
        .filter((d) => d.project >= 1 && d.project <= 5)
        // Sort by project in ascending order and timestamp in ascending order.
        .sort((a, b) => (a.project !== b.project ? a.project - b.project : a.timestamp - b.timestamp));

      allRecords.forEach((record) => {
        if (!highLevels.includes(record.high_level)) {
          const found = labelData.find((d) => d.label === record.label);
          record.high_level = found ? found.high_level : "REVISION";
        }
      });

      // Default loading of "All" item data
      await selectProject("All");
    } catch (err) {
      error = `Failed to load data: ${err instanceof Error ? err.message : String(err)}`;
      console.error("Error loading data:", err);
    } finally {
      loading = false;
    }
  }

  // Filter the data based on the selected items, recalculate the STEP and plot the
  async function selectProject(proj: string) {
    loading = true;
    error = null;
    try {
      selectedProject = proj;
      const projectNum = projectMap[proj];
      if (projectNum === null) {
        // All
        records = allRecords.slice();
      } else {
        records = allRecords.filter((d) => d.project === projectNum);
      }

      records.forEach((d, i) => (d.step = i + 1));
      filteredRecords = records.slice();
      brushStepRange = null;

      highlightMode = "normal";
      highlightSource = null;
      highlightTarget = null;

      drawChart();
    } catch (err) {
      error = `Failed to select project data: ${err instanceof Error ? err.message : String(err)}`;
      console.error("Error selecting project data:", err);
    } finally {
      loading = false;
    }
  }

  function handleBrush(event: any) {
    if (!event.selection) {
      filteredRecords = records.slice();
      brushStepRange = null;
    } else {
      const [x0, x1] = event.selection;
      const step0 = Math.max(1, Math.round(xScaleStep.invert(x0)));
      const step1 = Math.min(records.length, Math.round(xScaleStep.invert(x1)));

      filteredRecords = records.filter((d) => d.step! >= step0 && d.step! <= step1);
      brushStepRange = [step0, step1];
    }
    updateChart();
    updateBrushLabel();
  }

  function updateBrushLabel() {
    const svg = d3.select(svgElement);
    svg.selectAll(".brush-label").remove();

    if (brushStepRange) {
      const [step0, step1] = brushStepRange;
      const x0 = xScaleStep(step0);
      const x1 = xScaleStep(step1);

      svg.select(".container")
        .append("text")
        .attr("class", "brush-label")
        .attr("x", (x0 + x1) / 2)
        .attr("y", config.height - config.margin.bottom - 10)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("fill", "#333")
        .style("font-weight", "600")
        .text(`Steps: ${step0} - ${step1}`);
    }
  }

  function drawTimeAxis(container: d3.Selection<SVGGElement, unknown, null, undefined>) {
    const maxStep = records.length;
    const innerWidth = config.width - config.margin.left - config.margin.right;

    xScaleStep = d3
      .scaleLinear()
      .domain([1, maxStep])
      .range([config.margin.left, config.margin.left + innerWidth])
      .nice();

    container.selectAll(".time-axis").remove();
    container.selectAll(".brush").remove();
    container.selectAll(".brush-label").remove();

    const axis = d3
      .axisBottom(xScaleStep)
      .ticks(Math.min(10, maxStep))
      .tickFormat((d) => `${d}`);

    container.selectAll(".time-axis-title").remove(); // 先清理旧的标题
    container.append("text")
      .attr("class", "time-axis-title")
      .attr("x", config.margin.left + innerWidth / 2)  // 居中
      .attr("y", config.height - config.margin.bottom + config.timeAxisHeight / 2 - 20) // 比时间轴上移20像素
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "600")
      .style("fill", "#333")
      .text("Transition Step Selection");  // 这里改成你想显示的标题

    container
      .append("g")
      .attr("class", "time-axis")
      .attr("transform", `translate(0,${config.height - config.margin.bottom + config.timeAxisHeight / 2 + 15})`)
      .call(axis)
      .selectAll("text")
      .style("text-anchor", "middle")
      .style("font-size", "12px");

    brush = d3
      .brushX()
      .extent([
        [config.margin.left, config.height - config.margin.bottom],
        [config.margin.left + innerWidth, config.height - config.margin.bottom + config.timeAxisHeight],
      ])
      .on("brush end", handleBrush);

    const brushG = container.append("g").attr("class", "brush").call(brush);

    if (brushStepRange) {
      const [step0, step1] = brushStepRange;
      const x0 = xScaleStep(step0);
      const x1 = xScaleStep(step1);
      brushG.call(brush.move, [x0, x1]);
    }
  }

  function getSortedLabels(filteredTransitions: { source: string; target: string }[]) {
    const uniqueLabelsSet = new Set<string>();
    filteredTransitions.forEach(d => {
      uniqueLabelsSet.add(d.source);
      uniqueLabelsSet.add(d.target);
    });
    const uniqueLabels = Array.from(uniqueLabelsSet).filter(Boolean);

    const labelToHL = new Map<string, string>();
    uniqueLabels.forEach(label => {
      const found = labelData.find(d => d.label === label);
      labelToHL.set(label, found ? found.high_level : "REVISION");
    });

    const grouped = new Map<string, string[]>();
    highLevels.forEach(hl => grouped.set(hl, []));
    uniqueLabels.forEach(label => {
      const hl = labelToHL.get(label) || "REVISION";
      grouped.get(hl)?.push(label);
    });

    highLevels.forEach(hl => {
      const arr = grouped.get(hl);
      if (arr) {
        arr.sort((a, b) => {
          const aIndex = labelData.findIndex(d => d.label === a);
          const bIndex = labelData.findIndex(d => d.label === b);
          return aIndex - bIndex;
        });
      }
    });

    let sortedLabels: string[] = [];
    highLevels.forEach(hl => {
      const arr = grouped.get(hl);
      if (arr) sortedLabels = sortedLabels.concat(arr);
    });

    return sortedLabels;
  }

  function updateChart() {
    if (!svgElement) return;

    const svg = d3.select(svgElement);
    const { width, height, margin, timeAxisHeight } = config;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom - timeAxisHeight;

    const container = svg.select(".container");

    if (!filteredRecords.length) {
      svg.selectAll(".links").remove();
      svg.selectAll(".source-nodes").remove();
      svg.selectAll(".target-nodes").remove();

      svg.selectAll(".hl-group-rect").remove();
      svg.selectAll(".hl-group-label-left").remove();
      svg.selectAll(".hl-group-label-right").remove();
      return;
    }

    const transitions: { source: string; target: string }[] = [];
    for (let i = 0; i < filteredRecords.length - 1; i++) {
      const sourceLabel = showHighLevel
        ? filteredRecords[i].high_level
        : filteredRecords[i].label;
      const targetLabel = showHighLevel
        ? filteredRecords[i + 1].high_level
        : filteredRecords[i + 1].label;

      if (sourceLabel && targetLabel) {
        transitions.push({ source: sourceLabel, target: targetLabel });
      }
    }

    const filteredTransitions = showSelfTransitions
      ? transitions
      : transitions.filter((d) => d.source !== d.target);

    if (filteredTransitions.length === 0) {
      svg.selectAll(".links").remove();
      svg.selectAll(".source-nodes").remove();
      svg.selectAll(".target-nodes").remove();

      svg.selectAll(".hl-group-rect").remove();
      svg.selectAll(".hl-group-label-left").remove();
      svg.selectAll(".hl-group-label-right").remove();
      return;
    }

    let currentLabels: string[] = [];

    if (showHighLevel) {
      currentLabels = highLevels;
    } else {
      currentLabels = getSortedLabels(filteredTransitions);
    }

    const transitionCounts = d3.rollup(
      filteredTransitions,
      (v) => v.length,
      (d) => d.source,
      (d) => d.target
    );

    const nodeOutTotals = new Map<string, number>();
    const nodeInTotals = new Map<string, number>();
    const nodeOutLinks = new Map<string, { target: string; count: number }[]>();

    transitionCounts.forEach((targets, source) => {
      let total = 0;
      const links: { target: string; count: number }[] = [];
      targets.forEach((count, target) => {
        total += count;
        links.push({ target, count });
        nodeInTotals.set(target, (nodeInTotals.get(target) || 0) + count);
      });
      nodeOutTotals.set(source, total);
      nodeOutLinks.set(source, links.sort((a, b) => a.target.localeCompare(b.target)));
    });

    const maxTotal = d3.max([
      d3.max(Array.from(nodeOutTotals.values())),
      d3.max(Array.from(nodeInTotals.values())),
    ]) || 1;

    const maxRectLength = config.minBandSize - 5;
    const scaleFactor = maxTotal > 0 ? maxRectLength / maxTotal : 1;

    const requiredHeight = currentLabels.length * config.minBandSize;
    const adjustedHeight = Math.max(innerHeight, requiredHeight);

    const yScale = d3
      .scaleBand()
      .domain(currentLabels)
      .range([0, adjustedHeight])
      .padding(config.nodePadding)
      .paddingOuter(0.1);

      svg.selectAll(".links").remove();
      svg.selectAll(".source-nodes").remove();
      svg.selectAll(".target-nodes").remove();

      svg.selectAll(".hl-group-rect").remove();
      svg.selectAll(".hl-group-label-left").remove();
      svg.selectAll(".hl-group-label-right").remove();

    // Draw merged background rectangles for high-level groups if not in high-level view
    if (!showHighLevel) {
      const hlGroups = new Map<string, {start: number, end: number}>();
      const labelPositions = new Map<string, number>();
      
      currentLabels.forEach(label => {
        const found = labelData.find(d => d.label === label);
        const hl = found ? found.high_level : "REVISION";
        const pos = yScale(label)!;
        const bandwidth = yScale.bandwidth();
        
        if (!hlGroups.has(hl)) {
          hlGroups.set(hl, {start: pos, end: pos + bandwidth});
        } else {
          const group = hlGroups.get(hl)!;
          group.start = Math.min(group.start, pos);
          group.end = Math.max(group.end, pos + bandwidth);
        }
        
        labelPositions.set(label, pos + bandwidth/2);
      });

      // Draw the left background rectangle
      const leftGroup = container.append("g").attr("class", "hl-group-rect")
        .selectAll(".hl-group-left")
        .data(Array.from(hlGroups.entries()))
        .join("rect")
        .attr("class", "hl-group-left")
        .attr("x", -config.nodeWidth - 212)
        .attr("y", d => d[1].start)
        .attr("width", 210)
        .attr("height", d => d[1].end - d[1].start)
        .attr("fill", d => highLevelColors[d[0] as keyof typeof highLevelColors] || "#ddd")
        .attr("opacity", 0.3)
        .attr("rx", 2)
        .attr("ry", 2);

      // Add left grouping text
      container.append("g").attr("class", "hl-group-label-left")
        .selectAll("text")
        .data(Array.from(hlGroups.entries()))
        .join("text")
        .attr("x", -config.nodeWidth - 140)
        .attr("y", d => (d[1].start + d[1].end) / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "14px")
        .style("fill", d => highLevelColors[d[0] as keyof typeof highLevelColors] || "#444")
        .text(d => d[0]);

      // Draw the right background rectangle
      const rightGroup = container.append("g").attr("class", "hl-group-rect")
        .selectAll(".hl-group-right")
        .data(Array.from(hlGroups.entries()))
        .join("rect")
        .attr("class", "hl-group-right")
        .attr("x", innerWidth + config.nodeWidth + 2)
        .attr("y", d => d[1].start)
        .attr("width", 210)
        .attr("height", d => d[1].end - d[1].start)
        .attr("fill", d => highLevelColors[d[0] as keyof typeof highLevelColors] || "#ddd")
        .attr("opacity", 0.3)
        .attr("rx", 2)
        .attr("ry", 2);

      // Add right-hand grouping text
      container.append("g").attr("class", "hl-group-label-right")
        .selectAll("text")
        .data(Array.from(hlGroups.entries()))
        .join("text")
        .attr("x", innerWidth + config.nodeWidth + 140)
        .attr("y", d => (d[1].start + d[1].end) / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "14px")
        .style("fill", d => highLevelColors[d[0] as keyof typeof highLevelColors] || "#444")
        .text(d => d[0]);

          }

    const sourceNodes = container.append("g").attr("class", "source-nodes");

    sourceNodes
      .selectAll(".source-node")
      .data(currentLabels)
      .join("g")
      .attr("class", "source-node")
      .attr("transform", (d) => `translate(0,${yScale(d)! + yScale.bandwidth() / 2})`)
      .each(function (d) {
        const node = d3.select(this);
        const total = nodeOutTotals.get(d) || 0;
        const rectLength = total * scaleFactor;

        let hl = "REVISION";
        if (showHighLevel) {
          hl = d;
        } else {
          const found = labelData.find((l) => l.label === d);
          hl = found ? found.high_level : "REVISION";
        }

        const color = showHighLevel
          ? highLevelColors[d as keyof typeof highLevelColors] || "#4E79A7"
          : labelData.find((l) => l.label === d)?.color || "#4E79A7";

        node
          .append("line")
          .attr("x1", 0)
          .attr("x2", 0)
          .attr("y1", -rectLength / 2)
          .attr("y2", rectLength / 2)
          .attr("stroke", color)
          .attr("stroke-width", 2);

        node
          .append("rect")
          .attr("x", -config.nodeWidth)
          .attr("width", config.nodeWidth)
          .attr("height", rectLength)
          .attr("y", -rectLength / 2)
          .attr("fill", color)
          .attr("opacity", 0.6);

        const textWidth = d.length * 7 + 10;
        const textHeight = 14;

        node
          .append("text")
          .attr("x", -config.nodeWidth - 5)
          .attr("y", 0)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .style("font-size", "10px")
          .style("fill", "#000")
          .text(d);
      });

    const targetNodes = container
      .append("g")
      .attr("class", "target-nodes")
      .attr("transform", `translate(${innerWidth},0)`);

    targetNodes
      .selectAll(".target-node")
      .data(currentLabels)
      .join("g")
      .attr("class", "target-node")
      .attr("transform", (d) => `translate(0,${yScale(d)! + yScale.bandwidth() / 2})`)
      .each(function (d) {
        const node = d3.select(this);
        const total = nodeInTotals.get(d) || 0;
        const rectLength = total * scaleFactor;

        node
          .append("line")
          .attr("x1", 0)
          .attr("x2", 0)
          .attr("y1", -rectLength / 2)
          .attr("y2", rectLength / 2)
          .attr("stroke", config.targetColor)
          .attr("stroke-width", 2);

        node
          .append("rect")
          .attr("x", 0)
          .attr("width", config.nodeWidth)
          .attr("height", rectLength)
          .attr("y", -rectLength / 2)
          .attr("fill", config.targetColor)
          .attr("opacity", 0.6);

        const textWidth = d.length * 7 + 10;
        const textHeight = 14;

        node
          .append("text")
          .attr("x", config.nodeWidth + 5)
          .attr("y", 0)
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "middle")
          .style("font-size", "10px")
          .style("fill", "#000")
          .text(d);
      });

    const links: {
      source: string;
      target: string;
      count: number;
      sourceY?: number;
      targetY?: number;
      width?: number;
      color?: string;
    }[] = [];

    nodeOutLinks.forEach((targetLinks, source) => {
      const total = nodeOutTotals.get(source) || 1;
      const scaledTotal = total * scaleFactor;
      let currentY = -scaledTotal / 2;

      targetLinks.forEach((link) => {
        const width = link.count * scaleFactor;

        const color = showHighLevel
          ? highLevelColors[source as keyof typeof highLevelColors] || "#4E79A7"
          : labelData.find((l) => l.label === source)?.color || "#4E79A7";

        links.push({
          source,
          target: link.target,
          count: link.count,
          sourceY: currentY + width / 2,
          width,
          color,
        });
        currentY += width;
      });
    });

    const targetPositions = new Map<string, number>();
    links.forEach((link) => {
      const target = link.target;
      const total = nodeInTotals.get(target) || 1;
      const scaledTotal = total * scaleFactor;
      const currentY = targetPositions.get(target) || -scaledTotal / 2;

      link.targetY = currentY + link.width!;
      targetPositions.set(target, currentY + link.width!);
    });

    const linkGroup = container.append("g").attr("class", "links");

    const highLevelLabelSet = new Map<string, Set<string>>();
    highLevels.forEach(hl => {
      highLevelLabelSet.set(
        hl,
        new Set(labelData.filter(l => l.high_level === hl).map(l => l.label))
      );
    });

    linkGroup
      .selectAll(".link")
      .data(links)
      .join("path")
      .attr("class", "link")
      .attr("d", (d) => {
        const sourceY = yScale(d.source)! + yScale.bandwidth() / 2 + (d.sourceY || 0);
        const targetY = yScale(d.target)! + yScale.bandwidth() / 2 + (d.targetY || 0);

        return `M0,${sourceY} 
                C${innerWidth * 0.3},${sourceY} 
                ${innerWidth * 0.7},${targetY} 
                ${innerWidth},${targetY}`;
      })
      .attr("fill", "none")
      .attr("stroke", (d) => {
        if (highlightMode === "normal") {
          if (highlightSource && highlightTarget) {
            return (d.source === highlightSource && d.target === highlightTarget) ? d.color! : "#eee";
          } else if (highlightSource) {
            return d.source === highlightSource ? d.color! : "#eee";
          } else if (highlightTarget) {
            return d.target === highlightTarget ? config.targetColor : "#eee";
          } else {
            return d.color!;
          }
        } else {
          let sourceHLs = highlightSource ? highLevelLabelSet.get(highlightSource) : null;
          let targetHLs = highlightTarget ? highLevelLabelSet.get(highlightTarget) : null;

          let sourceMatch = sourceHLs ? sourceHLs.has(d.source) : false;
          let targetMatch = targetHLs ? targetHLs.has(d.target) : false;

          if (highlightSource && highlightTarget) {
            return (sourceMatch && targetMatch) ? d.color! : "#eee";
          } else if (highlightSource) {
            return sourceMatch ? d.color! : "#eee";
          } else if (highlightTarget) {
            return targetMatch ? config.targetColor : "#eee";
          } else {
            return d.color!;
          }
        }
      })
      .attr("stroke-width", (d) => d.width!)
      .attr("opacity", (d) => {
        if (highlightMode === "normal") {
          if (
            (highlightSource && d.source !== highlightSource) ||
            (highlightTarget && d.target !== highlightTarget)
          )
            return 0.2;
          return 0.6;
        } else {
          let sourceHLs = highlightSource ? highLevelLabelSet.get(highlightSource) : null;
          let targetHLs = highlightTarget ? highLevelLabelSet.get(highlightTarget) : null;

          let sourceMatch = sourceHLs ? sourceHLs.has(d.source) : false;
          let targetMatch = targetHLs ? targetHLs.has(d.target) : false;

          if (highlightSource && highlightTarget) {
            return sourceMatch && targetMatch ? 0.9 : 0.2;
          } else if (highlightSource) {
            return sourceMatch ? 0.9 : 0.2;
          } else if (highlightTarget) {
            return targetMatch ? 0.9 : 0.2;
          } else {
            return 0.6;
          }
        }
      })
      .on("mouseover", function (event: MouseEvent, d) {
        d3.select(this).raise().attr("opacity", 0.9);
        const tooltip = d3.select("body").select(".tooltip");
        tooltip
          .style("opacity", 1)
          .html(`${d.source} → ${d.target}<br/>Count: ${d.count}`)
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 28}px`);
      })
      .on("mouseout", function () {
        d3.select(this).attr("opacity", (d) => {
          if (highlightMode === "normal") {
            if (
              (highlightSource && d.source !== highlightSource) ||
              (highlightTarget && d.target !== highlightTarget)
            )
              return 0.2;
            return 0.6;
          } else {
            let sourceHLs = highlightSource ? highLevelLabelSet.get(highlightSource) : null;
            let targetHLs = highlightTarget ? highLevelLabelSet.get(highlightTarget) : null;

            let sourceMatch = sourceHLs ? sourceHLs.has(d.source) : false;
            let targetMatch = targetHLs ? targetHLs.has(d.target) : false;

            if (highlightSource && highlightTarget) {
              return sourceMatch && targetMatch ? 0.9 : 0.2;
            } else if (highlightSource) {
              return sourceMatch ? 0.9 : 0.2;
            } else if (highlightTarget) {
              return targetMatch ? 0.9 : 0.2;
            } else {
              return 0.6;
            }
          }
        });
        d3.select("body").select(".tooltip").style("opacity", 0);
      });

    if (highlightSource || highlightTarget) {
      linkGroup
        .selectAll(".link")
        .sort((a, b) => {
          const aHighlighted =
            highlightMode === "normal"
              ? (highlightSource && a.source === highlightSource) ||
                (highlightTarget && a.target === highlightTarget)
              : (highlightSource &&
                  highLevelLabelSet.get(highlightSource)?.has(a.source)) ||
                (highlightTarget &&
                  highLevelLabelSet.get(highlightTarget)?.has(a.target));
          const bHighlighted =
            highlightMode === "normal"
              ? (highlightSource && b.source === highlightSource) ||
                (highlightTarget && b.target === highlightTarget)
              : (highlightSource &&
                  highLevelLabelSet.get(highlightSource)?.has(b.source)) ||
                (highlightTarget &&
                  highLevelLabelSet.get(highlightTarget)?.has(b.target));
          return aHighlighted === bHighlighted ? 0 : aHighlighted ? 1 : -1;
        });
    }
  }

function drawChart() {
  if (!svgElement) return;
  const svg = d3.select(svgElement);
  svg.selectAll("*").remove();

  const { width, height, margin } = config;

  svg.attr("viewBox", `0 0 ${width} ${height}`)
     .attr("preserveAspectRatio", "xMidYMid meet");

  const container = svg.append("g")
    .attr("class", "container")
    .attr("transform", "translate(0, 60)");
  drawTimeAxis(container);
  updateChart();

  svg
    .append("text")
    .attr("x", width / 2)
    .attr("y", 30)
    .attr("text-anchor", "middle")
    .style("font-size", "16px")
    .style("font-weight", "bold")
    .text(
      `Label Transition Flow - ${selectedProject} (${showHighLevel ? "High Level" : "Detailed"})`
    );
}


  function toggleSelfTransitions() {
    showSelfTransitions = !showSelfTransitions;
    updateChart();
  }

  function toggleHighLevel() {
    showHighLevel = !showHighLevel;
    highlightMode = "normal";
    highlightSource = null;
    highlightTarget = null;
    if (filteredRecords.length === 0) {
      filteredRecords = records.slice();
      brushStepRange = null;
    }
    drawChart();
  }

  function toggleHighlightMode() {
    if (highlightMode === "normal") {
      highlightMode = "highLevelHighlight";
      highlightSource = null;
      highlightTarget = null;
    } else {
      highlightMode = "normal";
      highlightSource = null;
      highlightTarget = null;
    }
    updateChart();
  }

  function updateHighlightSource(event: Event) {
    const select = event.target as HTMLSelectElement;
    const val = select.value;
    highlightSource = val === "All" ? null : val;
    updateChart();
  }

  function updateHighlightTarget(event: Event) {
    const select = event.target as HTMLSelectElement;
    const val = select.value;
    highlightTarget = val === "All" ? null : val;
    updateChart();
  }

  function onProjectChange(event: Event) {
    const select = event.target as HTMLSelectElement;
    selectProject(select.value);
  }

onMount(async () => {
  d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);
  await loadAllData();
  if (svgElement) {
    drawChart();
  }
});

</script>

<style>
  svg {
    display: block;
    margin: 20px auto;
    background: white;
    border-radius: 0;
    box-shadow: none;
    overflow: visible;
    max-height: 80vh;
    width: 100%;
  }

  .tooltip {
    position: absolute;
    pointer-events: none;
    opacity: 0;
    padding: 8px;
    font-size: 12px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    border-radius: 4px;
    transition: opacity 0.2s;
    z-index: 100;
  }

  .controls {
    display: flex;
    margin: 15px auto;
    width: 900px;
    flex-wrap: wrap;
    gap: 20px;
    font-family: Arial, sans-serif;
  }

  .control-section {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }

  .control-section label {
    font-weight: 600;
    font-size: 14px;
    margin-right: 10px;
  }

  .control-section select,
  .control-section button {
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid #ccc;
    cursor: pointer;
  }

  .control-section button {
    background: #4e79a7;
    color: white;
    border: none;
    transition: background-color 0.3s;
  }

  .control-section button:hover {
    background: #3a5f8a;
  }

  /* Project selector section */
  .project-section {
    flex-grow: 1;
    min-width: 160px;
  }

  /* Highlight controls section */
  .highlight-section {
    flex-grow: 3;
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
  }

  .highlight-subsection {
    display: flex;
    flex-direction: column;
  }

  .highlight-subsection label {
    font-weight: 600;
    margin-bottom: 4px;
  }

  /* Buttons group */
  .buttons-section {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .loading,
  .error {
    text-align: center;
    padding: 50px;
    font-size: 18px;
  }

  .loading {
    color: #666;
  }

  .error {
    color: #e15759;
  }

  .brush .selection {
    fill: steelblue;
    fill-opacity: 0.3;
    stroke: #333;
    stroke-width: 1.5px;
    stroke-dasharray: 3, 3;
  }

  .brush .handle {
    fill: #333;
    stroke: #fff;
    stroke-width: 1.5px;
  }
</style>

<h1>Label Transition Flow Visualization with Step Brush</h1>

{#if error}
  <div class="error">{error}</div>
{:else if loading}
  <div class="loading">Loading and processing data, please wait...</div>
{:else}
  <div class="controls">
    <!-- Project selector -->
    <div class="control-section project-section" aria-label="Project selection">
      <label for="project-select">Project:</label>
      <select id="project-select" on:change={onProjectChange} bind:value={selectedProject}>
        {#each projects as p}
          <option value={p} selected={selectedProject === p}>{p}</option>
        {/each}
      </select>
    </div>

    <!-- Highlight controls -->
    <div class="control-section highlight-section" aria-label="Highlight controls">
      <div class="highlight-subsection">
        {#if !showHighLevel}
          <button on:click={toggleHighlightMode} aria-pressed={highlightMode !== "normal"}>
            {highlightMode === "normal" ? "Switch to High Level Highlight" : "Switch to Normal Highlight"}
          </button>
        {/if}
      </div>

      <div class="highlight-subsection">
        <label for="highlight-source-select">Source:</label>
        <select id="highlight-source-select" on:change={updateHighlightSource}>
          <option value="All">All</option>
          {#if showHighLevel}
            {#each highLevels as hl}
              <option value={hl} selected={highlightSource === hl}>{hl}</option>
            {/each}
          {:else}
            {#if highlightMode === "normal"}
              {#each labelData.map((d) => d.label) as label}
                <option value={label} selected={highlightSource === label}>{label}</option>
              {/each}
            {:else}
              {#each highLevels as hl}
                <option value={hl} selected={highlightSource === hl}>{hl}</option>
              {/each}
            {/if}
          {/if}
        </select>
      </div>

      <div class="highlight-subsection">
        <label for="highlight-target-select">Target:</label>
        <select id="highlight-target-select" on:change={updateHighlightTarget}>
          <option value="All">All</option>
          {#if showHighLevel}
            {#each highLevels as hl}
              <option value={hl} selected={highlightTarget === hl}>{hl}</option>
            {/each}
          {:else}
            {#if highlightMode === "normal"}
              {#each labelData.map((d) => d.label) as label}
                <option value={label} selected={highlightTarget === label}>{label}</option>
              {/each}
            {:else}
              {#each highLevels as hl}
                <option value={hl} selected={highlightTarget === hl}>{hl}</option>
              {/each}
            {/if}
          {/if}
        </select>
      </div>
    </div>

    <!-- Other controls -->
    <div class="control-section buttons-section" aria-label="Other controls">
      <button on:click={toggleSelfTransitions} aria-pressed={showSelfTransitions}>
        {showSelfTransitions ? "Hide Self" : "Show Self"}
      </button>

      <button on:click={toggleHighLevel} aria-pressed={showHighLevel}>
        {showHighLevel ? "Detailed View" : "High Level View"}
      </button>
    </div>
  </div>

  <svg width={config.width} height={config.height} bind:this={svgElement}></svg>
{/if}