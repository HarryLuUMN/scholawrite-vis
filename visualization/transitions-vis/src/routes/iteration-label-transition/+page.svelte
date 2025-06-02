<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { generateLabelMap } from '$lib/labelMap';
  import { goto } from '$app/navigation';

  export let data: { models: string[] };

  let models: string[] = data.models;
  let selectedModel: string = models[0];
  let svgContainer: SVGSVGElement;
  let labelMap: Record<string, { short: string; color: string }> = {};

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
  });

  async function loadAndRender() {
    const url = `/data/iteration-label-transition/${selectedModel}.csv`;

    const res = await fetch(url);
    if (!res.ok) {
      console.error(`Fetch failed for: ${url}`);
      return;
    }

    const text = await res.text();
    const parsed: LabelRow[] = d3.csvParse(text);

    if (!parsed || parsed.length === 0 || !parsed[0].label) {
      console.error('Parsed data invalid or missing label field:', parsed);
      return;
    }

    labelMap = generateLabelMap(parsed);
    drawGrid(parsed, labelMap);
  }

  function drawGrid(data: LabelRow[], labelMap: Record<string, { short: string; color: string }>) {
    const cellSize = 40;
    const paddingLeft = 100;
    const paddingTop = 30;

    const iterations = data.map(d => +d.iteration);
    const maxIter = d3.max(iterations) ?? 0;

    const width = paddingLeft + (maxIter + 1) * cellSize;
    const height = cellSize + paddingTop;

    d3.select(svgContainer).selectAll('*').remove();

    const svg = d3.select(svgContainer)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g');

    // iteration numbers
    for (let i = 0; i <= maxIter; i++) {
      g.append('text')
        .attr('x', paddingLeft + i * cellSize + cellSize / 2)
        .attr('y', paddingTop - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', '#444')
        .text(i);
    }

    // project label
    g.append('text')
      .attr('x', paddingLeft - 10)
      .attr('y', paddingTop + cellSize / 2 + 5)
      .attr('text-anchor', 'end')
      .attr('font-size', '12px')
      .text('Project 1');

    // colored cells
    g.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => paddingLeft + +d.iteration * cellSize)
      .attr('y', paddingTop)
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', d => labelMap[d.label]?.color || '#ccc')
      .attr('stroke', '#999');

    // short label in cells
    g.selectAll('text.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => paddingLeft + +d.iteration * cellSize + cellSize / 2)
      .attr('y', paddingTop + cellSize / 2 + 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .text(d => labelMap[d.label]?.short || '?');
  }

  function onModelChange(event: Event) {
    selectedModel = (event.target as HTMLSelectElement).value;
    loadAndRender();
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
