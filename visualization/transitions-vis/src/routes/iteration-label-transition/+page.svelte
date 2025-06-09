<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { generateLabelMap } from '$lib/labelMap';
  import { goto } from '$app/navigation';
  import { categoryBackgroundColors, hierarchicalLabel } from './const';
    import { base } from '$app/paths';

  export let data: { models: string[] };

  let models: string[] = data.models;
  let selectedModel: string = models[0];
  let svgContainer: SVGSVGElement;
  let labelMap: Record<string, { short: string; color: string }> = {};
  let viewState = 'detailed'; // 'high-level' or 'detailed';
  let parsed:LabelRow[] = [];
  let zoomScale = 1.0;

  const baseCellSize = 40;
  const viewStates = ['high-level', 'detailed'];
  const cellSize = baseCellSize * zoomScale;

  interface LabelRow {
    iteration: string;
    label: string;
  }

  function goBack() {
    goto('/');
  }

  onMount(() => {
    if (selectedModel) {
      loadAndRender();
    }

    const svg = d3.select(svgContainer);
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 5])
        .on('zoom', (event) => {
          const newScale = event.transform.k;
          if (Math.abs(zoomScale - newScale) > 0.01) {
            zoomScale = newScale;
            redraw();
          }
        })
    );
  });

  async function loadAndRender() {
    const url = `/data/iteration-label-transition/${selectedModel}.csv`;

    const res = await fetch(url);
    if (!res.ok) {
      console.error(`Fetch failed for: ${url}`);
      return;
    }

    const text = await res.text();
    parsed = d3.csvParse(text);

    if (!parsed || parsed.length === 0 || !parsed[0].label) {
      console.error('Parsed data invalid or missing label field:', parsed);
      return;
    }

    labelMap = generateLabelMap(parsed);
    if (viewState === 'high-level') {
      drawHighLevelGrid(parsed);
    } else {
      drawDetailedLevelGrid(parsed);
    }
  }

  function drawHighLevelGrid(parsed: LabelRow[]) {
    let i = 0;

    const iterations = parsed.map(d => +d.iteration);
  const maxIter = d3.max(iterations) ?? 0;

  const paddingLeft = 100;
  const width = paddingLeft + (maxIter + 1) * getCellSize();
  
  const height = 3 * getCellSize() + 60;
    
    d3.select(svgContainer).selectAll('*').remove();

      d3.select(svgContainer)
        .attr('width', width)
        .attr('height', height)
        .attr("id", "iteration-label-grid");
    for (const [category, labels] of Object.entries(hierarchicalLabel)) {
      let iterLabels = false;
      if (i === 0) iterLabels = true;
      drawGrid(parsed, 100, 30 + i * getCellSize(), labelMap, category, iterLabels);
      i++;
    }

  }

  function drawDetailedLevelGrid(parsed: LabelRow[]) {
  const allDetailedLabels = Object.values(hierarchicalLabel).flat();
  d3.select(svgContainer).selectAll('*').remove();

  const iterations = parsed.map(d => +d.iteration);
  const maxIter = d3.max(iterations) ?? 0;

  const paddingLeft = 100;
  const width = paddingLeft + (maxIter + 1) * getCellSize();
  
  const height = allDetailedLabels.length * getCellSize() + 60;
  console.log("len", allDetailedLabels.length, height);

  d3.select(svgContainer)
    .attr('width', width)
    .attr('height', height)
    .attr("id", "detailed-label-grid");

  allDetailedLabels.forEach((label, i) => {
    const pseudoCategory = label;
    const pseudoHierarchy: Record<string, string[]> = { [label]: [label] };
    drawGrid(
      parsed,
      paddingLeft,
      30 + i * getCellSize(),
      labelMap,
      pseudoCategory,
      i === 0,
      pseudoHierarchy
    )
  });
}

  function drawGrid(
    data: LabelRow[], 
    startX: number,
    startY: number,
    labelMap: Record<string, { short: string; color: string }>,
    category: string,
    iterLabels: boolean = true,
    labelFilterMap: Record<string, string[]> = hierarchicalLabel
  ) {
    const paddingLeft = startX;
    const paddingTop = startY;

    const iterations = data.map(d => +d.iteration);
    const maxIter = d3.max(iterations) ?? 0;

    const width = paddingLeft + (maxIter + 1) * getCellSize();
    const height = getCellSize() + paddingTop;

    const svg = d3.select(svgContainer);
    const g = svg.append('g');

    const bg = svg.append("g").attr("class", "bg")


    const rowWidth = paddingLeft + (maxIter + 1) * getCellSize();

    g.append('rect')
      .attr('x', 0)
      .attr('y', paddingTop)
      .attr('width', rowWidth)
      .attr('height', getCellSize())
      .attr('fill', categoryBackgroundColors[resolveHighLevel(category)] || '#ffffff');


    if(iterLabels && getCellSize() >= baseCellSize/2) {
      for (let i = 0; i <= maxIter; i++) {
        g.append('text')
          .attr('x', paddingLeft + i * getCellSize() + getCellSize() / 2)
          .attr('y', paddingTop - 10)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#444')
          .text(i);
      }
    }

    g.append('text')
      .attr('x', paddingLeft - 10)
      .attr('y', paddingTop + getCellSize() / 2 + 5)
      .attr('text-anchor', 'end')
      .attr('font-size', '12px')
      .text(category);



    const filteredData = data.filter((d:any) => labelFilterMap[category]?.includes(d.label));

    g.selectAll('rect.cell')
      .data(filteredData)
      .enter()
      .append('rect')
      .attr('x', (d: any) => paddingLeft + +d.iteration * getCellSize())
      .attr('y', paddingTop)
      .attr('width', getCellSize())
      .attr('height', getCellSize())
      .attr('fill', (d: any) => labelMap[d.label]?.color || '#ccc')
      .attr('stroke', '#999');

    g.selectAll('circle')
      .data(filteredData)
      .enter()
      .append('circle')
      .attr('cx', (d: any) => paddingLeft + +d.iteration * getCellSize() + getCellSize() / 2)
      .attr('cy', paddingTop + getCellSize() / 2)
      .attr('r', getCellSize() / 2)
      .attr('fill', "none")
      .attr('stroke', 'black');

    g.selectAll('text.label')
      .data(filteredData)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', (d: any) => paddingLeft + +d.iteration * getCellSize() + getCellSize() / 2)
      .attr('y', paddingTop + getCellSize() / 2 + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text((d: any) => labelMap[d.label]?.short || '?');
  }

  function resolveHighLevel(label: string): string {
    for (const [highLevel, subLabels] of Object.entries(hierarchicalLabel)) {
      if (subLabels.includes(label) || highLevel === label) return highLevel;
    }
    return 'unknown';
  }


  function getCellSize() {
    return baseCellSize * zoomScale;
  }

  function redraw() {
    d3.select(svgContainer).selectAll('*').remove();
    if (viewState === 'high-level') {
      drawHighLevelGrid(parsed);
    } else {
      drawDetailedLevelGrid(parsed);
    }
  }


  function onModelChange(event: Event) {
    selectedModel = (event.target as HTMLSelectElement).value;
    loadAndRender();
  }

  function onViewChange(event: Event) {
    viewState = (event.target as HTMLSelectElement).value;
    if (viewState === 'high-level') {
      drawHighLevelGrid(parsed);
    } else {
      drawDetailedLevelGrid(parsed);
    }
  }
</script>

<button class="back-button" on:click={goBack}>← Dashboard</button>

<main>
  <h1>Iteration Label Grid</h1>

  <div class="control-panel">
    <label for="model-select">Select Model:</label>
    <select id="model-select" bind:value={selectedModel} on:change={onModelChange}>
      {#each models as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
  </div>
  <div class="control-panel">
    <label for="model-select">Select View:</label>
    <select id="model-select" bind:value={viewState} on:change={onViewChange}>
      {#each viewStates as state}
        <option value={state}>{state}</option>
      {/each}
    </select>
  </div>

  <div class="legend">
    <h3>Legend</h3>
    <ul>
      {#each Object.entries(labelMap) as [label, info]}
        <li><span style="background-color: {info.color}"></span>{info.short} = {label}</li>
      {/each}
    </ul>
  </div>

  <div class="scroll-container">
    <svg bind:this={svgContainer}></svg>
  </div>
</main>

<style>
  main {
    padding: 80px 40px 40px 40px;
  }

  h1 {
    color: #333;
  }

  .control-panel {
    margin-bottom: 16px;
  }

  select {
    font-size: 1rem;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #aaa;
    background: #f8f8f8;
    margin-left: 10px;
  }

  .legend {
    margin-bottom: 16px;
    background: #f9f9f9;
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  }

  .legend h3 {
    margin-top: 0;
    font-size: 14px;
    margin-bottom: 8px;
  }

  .legend ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .legend li {
    font-size: 13px;
    display: flex;
    align-items: center;
  }

  .legend li span {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    margin-right: 6px;
  }

  .scroll-container {
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #ccc;
    padding-bottom: 10px;
    width: 100%;
  }

  svg {
    display: block;
  }

  .back-button {
    position: fixed;
    top: 12px;
    left: 12px;
    padding: 8px 14px;
    background-color: #4e79a7;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(78,121,167,0.4);
    font-size: 0.9rem;
    transition: background-color 0.2s ease;
    z-index: 1000;
  }

  .back-button:hover {
    background-color: #3a5f8a;
  }
</style>
