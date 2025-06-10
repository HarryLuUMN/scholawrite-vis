<script lang="ts">
  import { onMount } from 'svelte';

  let models: string[] = [];
  let selectedModel = '';
  let iterations: string[] = [];
  let selectedBefore = '';
  let selectedAfter = '';
  let beforeText = '';
  let afterText = '';
  let highlightedAfter = '';
  let changeDiffs: { index: number; diffSize: number }[] = [];
  let jumpLinks: { from: number; to: number; diffSize: number }[] = [];

  let selectedBeforeIndex: number | null = null;
  let selectedAfterIndex: number | null = null;

  onMount(async () => {
    const res = await fetch('/data/text-comparison/models.json');
    models = await res.json();
    if (models.length > 0) {
      selectedModel = models[0];
      await loadIterations();
    }
  });

  async function loadIterations() {
    const res = await fetch(`/data/text-comparison/${selectedModel}/files.json`);
    const rawList = await res.json();
    iterations = rawList.sort((a, b) => {
      const extract = (s: string) => parseInt(s.match(/\d+/)?.[0] || '0', 10);
      return extract(a) - extract(b);
    });

    selectedBeforeIndex = 0;
    selectedAfterIndex = 1;
    selectedBefore = iterations[0];
    selectedAfter = iterations[1] || iterations[0];
    await computeAllDiffs();
    computeJumpLinks();
    await loadTexts();
  }

  async function loadTexts() {
    if (!selectedModel || selectedBeforeIndex === null || selectedAfterIndex === null) return;
    const [beforeRes, afterRes] = await Promise.all([
      fetch(`/data/text-comparison/${selectedModel}/${selectedBefore}`),
      fetch(`/data/text-comparison/${selectedModel}/${selectedAfter}`)
    ]);
    beforeText = await beforeRes.text();
    afterText = await afterRes.text();
    highlightDifferences();
  }

  function escapeHtml(text: string) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlightDifferences() {
    const beforeLines = beforeText.split('\n');
    const afterLines = afterText.split('\n');
    highlightedAfter = afterLines.map((line, i) => {
      const safeLine = escapeHtml(line);
      return safeLine === escapeHtml(beforeLines[i] || '') ? safeLine : `<mark>${safeLine}</mark>`;
    }).join('\n');
  }

  async function computeAllDiffs() {
    changeDiffs = [];
    for (let i = 0; i < iterations.length - 1; i++) {
      const [resA, resB] = await Promise.all([
        fetch(`/data/text-comparison/${selectedModel}/${iterations[i]}`),
        fetch(`/data/text-comparison/${selectedModel}/${iterations[i + 1]}`)
      ]);
      const textA = await resA.text();
      const textB = await resB.text();
      let diffCount = 0;
      const maxLen = Math.max(textA.length, textB.length);
      for (let j = 0; j < maxLen; j++) {
        if (textA[j] !== textB[j]) diffCount++;
      }
      changeDiffs.push({ index: i, diffSize: diffCount });
    }
  }

  function computeJumpLinks() {
    jumpLinks = changeDiffs.map(d => ({
      from: d.index,
      to: d.index + 1,
      diffSize: d.diffSize
    }));
  }

  function getCurveColor(diffSize: number) {
    if (diffSize > 800) return '#8B0000';
    if (diffSize > 500) return '#CC3300';
    if (diffSize > 300) return '#FF6600';
    if (diffSize > 150) return '#FFCC00';
    if (diffSize > 50) return '#99CC33';
    return '#CCCCCC';
  }

  function selectIteration(index: number) {
    if (selectedBeforeIndex === null || (selectedBeforeIndex !== null && selectedAfterIndex !== null)) {
      selectedBeforeIndex = index;
      selectedAfterIndex = null;
      selectedBefore = iterations[index];
    } else {
      if (index <= selectedBeforeIndex) {
        alert('After iteration must be after Before iteration.');
        return;
      }
      selectedAfterIndex = index;
      selectedAfter = iterations[index];
      loadTexts();
    }
  }

  function resetSelection() {
    selectedBeforeIndex = null;
    selectedAfterIndex = null;
    selectedBefore = '';
    selectedAfter = '';
    // beforeText = '';
    // afterText = '';
    // highlightedAfter = '';
  }

  function isBetween(i: number) {
    return selectedBeforeIndex !== null &&
           selectedAfterIndex !== null &&
           i >= selectedBeforeIndex && i < selectedAfterIndex;
  }
</script>

<style>
  .controls {
    display: flex;
    gap: 1rem;
    margin: 1rem 2rem;
    flex-wrap: wrap;
  }

  .text-container {
    display: flex;
    gap: 2rem;
    margin: 1rem 2rem;
    flex-wrap: wrap;
  }

  .text-box {
    flex: 1;
    min-width: 400px;
    padding: 1rem;
    background: #fcfcfc;
    border: 1px solid #ccc;
    font-family: monospace;
    white-space: pre-wrap;
    overflow-y: auto;
    height: 600px;
    border-radius: 8px;
  }

  mark {
    background: yellow;
  }

  .graph-wrapper {
    position: relative;
    margin: 2rem;
    background: #fafafa;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    height: 240px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .svg-scroll-container {
    overflow-x: auto;
    overflow-y: hidden;
    flex-grow: 1;
  }

  .iteration-graph {
    height: 200px;
    display: block;
  }

  .reset-btn {
    position: absolute;
    bottom: 10px;
    right: 10px;
    background: #eee;
    border: 1px solid #999;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 0.8rem;
    cursor: pointer;
    z-index: 1;
  }

  .legend {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    margin-top: 0.5rem;
    margin-left: 0.5rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
  }

  .legend-box {
    width: 16px;
    height: 16px;
    margin-right: 4px;
  }

  .blink {
    animation: blink 1s linear infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
</style>

<div class="controls">
  <label>Model:
    <select bind:value={selectedModel} on:change={loadIterations}>
      {#each models as m}
        <option value={m}>{m}</option>
      {/each}
    </select>
  </label>
</div>

{#if beforeText && highlightedAfter}
  <div class="text-container">
    <div class="text-box">
      <strong>Before Iteration: {selectedBefore}</strong>
      <pre>{beforeText}</pre>
    </div>
    <div class="text-box">
      <strong>After Iteration: {selectedAfter}</strong>
      <pre>{@html highlightedAfter}</pre>
    </div>
  </div>
{/if}

{#if jumpLinks.length}
  <div class="graph-wrapper">
    <div class="svg-scroll-container">
      <svg
        class="iteration-graph"
        viewBox={`0 0 ${iterations.length * 80 + 100} 200`}
        preserveAspectRatio="xMinYMid meet"
      >
        {#each iterations as iter, i}
          <g transform="translate({i * 80 + 50}, 60)" on:click={() => selectIteration(i)} style="cursor:pointer">
            <circle
              r="10"
              fill={(i === selectedBeforeIndex || i === selectedAfterIndex) ? '#f00' : '#4e79a7'}
              class={(i === selectedBeforeIndex && selectedAfterIndex === null) ? 'blink' : ''}
            />
            <text y="30" font-size="14" text-anchor="middle">{`i${i}`}</text>
          </g>

          {#if i < iterations.length - 1}
            <line
              x1={i * 80 + 50}
              y1={60}
              x2={(i + 1) * 80 + 50}
              y2={60}
              stroke={isBetween(i) ? '#f00' : '#999'}
              stroke-width={isBetween(i) ? 5 : 2}
            />
          {/if}
        {/each}

        {#each jumpLinks as link}
          <path
            d={`M ${link.from * 80 + 50},60 Q ${(link.from + link.to) * 40},20 ${link.to * 80 + 50},60`}
            fill="none"
            stroke={getCurveColor(link.diffSize)}
            stroke-width={2 + Math.min(link.diffSize / 100, 5)}
          />
        {/each}
      </svg>
    </div>

    <div class="legend">
      <div class="legend-item"><span class="legend-box" style="background:#CCCCCC;"></span> 0–50</div>
      <div class="legend-item"><span class="legend-box" style="background:#99CC33;"></span> 50–150</div>
      <div class="legend-item"><span class="legend-box" style="background:#FFCC00;"></span> 150–300</div>
      <div class="legend-item"><span class="legend-box" style="background:#FF6600;"></span> 300–500</div>
      <div class="legend-item"><span class="legend-box" style="background:#CC3300;"></span> 500–800</div>
      <div class="legend-item"><span class="legend-box" style="background:#8B0000;"></span> >800</div>
    </div>

    <button class="reset-btn" on:click={resetSelection}>Reset</button>
  </div>
{/if}
