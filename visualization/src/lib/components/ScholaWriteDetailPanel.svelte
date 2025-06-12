<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { scholawrite_train_DATA_URL } from '$lib/paths';


  let data: any[] = [];

  onMount(async () => {
    const raw = await d3.csv(scholawrite_train_DATA_URL);
    data = raw.map(d => ({ ...d, timestamp: +d.timestamp }));
    drawLabelBarChart();
    drawTimeLineChart();
    drawAuthorPieChart();
    drawHighLevelBarChart();
  });

  function drawLabelBarChart() {
    const container = d3.select('#label-bar');
    container.selectAll('*').remove();

    const counts = d3.rollup(data, v => v.length, d => d.label);
    const entries = Array.from(counts, ([label, count]) => ({ label, count }));

    const width = 500, height = 300, margin = { top: 30, right: 20, bottom: 100, left: 60 };
    const svg = container.append('svg').attr('width', width).attr('height', height);
    const x = d3.scaleBand().domain(entries.map(d => d.label)).range([margin.left, width - margin.right]).padding(0.2);
    const y = d3.scaleLinear().domain([0, d3.max(entries, d => d.count)!]).nice().range([height - margin.bottom, margin.top]);

    const tooltip = container.append("div").attr("class", "tooltip").style("opacity", 0);

    svg.selectAll('rect')
      .data(entries)
      .enter()
      .append('rect')
      .attr('x', d => x(d.label)!)
      .attr('y', d => y(d.count))
      .attr('width', x.bandwidth())
      .attr('height', d => y(0) - y(d.count))
      .attr('fill', '#3498db')
      .on('mouseover', (e, d) => {
        tooltip.transition().duration(200).style("opacity", .9);
        tooltip.html(`${d.label}: ${d.count}`)
               .style("left", (e.pageX + 10) + "px")
               .style("top", (e.pageY - 28) + "px");
      })
      .on('mouseout', () => tooltip.transition().duration(300).style("opacity", 0));

    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end");

    svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y));
  }

  function drawTimeLineChart() {
    const container = d3.select('#time-line');
    container.selectAll('*').remove();

    const timeBins = d3.rollup(data, v => v.length, d => d3.timeDay(new Date(+d.timestamp)));
    const entries = Array.from(timeBins, ([date, count]) => ({ date, count })).sort((a, b) => +a.date - +b.date);

    const width = 500, height = 300, margin = { top: 30, right: 20, bottom: 40, left: 60 };
    const svg = container.append('svg').attr('width', width).attr('height', height);
    const x = d3.scaleTime().domain(d3.extent(entries, d => d.date) as [Date, Date]).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([0, d3.max(entries, d => d.count)!]).nice().range([height - margin.bottom, margin.top]);

    svg.append('path')
      .datum(entries)
      .attr('fill', 'none')
      .attr('stroke', '#e67e22')
      .attr('stroke-width', 2)
      .attr('d', d3.line<any>().x(d => x(d.date)).y(d => y(d.count)));

    svg.append('g').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(6));
    svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y));
  }

  function drawAuthorPieChart() {
  const container = d3.select('#author-pie');
  container.selectAll('*').remove();

  const counts = d3.rollup(data, v => v.length, d => d.author);
  const entries = Array.from(counts.entries());

  const width = 500, height = 320, radius = Math.min(width, height) / 2 - 40;
  const svg = container.append('svg').attr('width', width).attr('height', height);
  const g = svg.append('g').attr('transform', `translate(${width / 2}, ${height / 2})`);

  const color = d3.scaleOrdinal(d3.schemeCategory10);
  const pie = d3.pie().value(d => d[1])(entries);
  const arc = d3.arc<d3.PieArcDatum<[string, number]>>().outerRadius(radius).innerRadius(0);

  // Hover display element
  const hoverText = container.append('div')
    .attr('class', 'chart-legend-info')
    .style('position', 'absolute')
    .style('left', '10px')
    .style('bottom', '10px')
    .style('font-size', '13px')
    .style('color', '#2c3e50')
    .text('');

  g.selectAll('path')
    .data(pie)
    .enter()
    .append('path')
    .attr('d', arc)
    .attr('fill', d => color(d.data[0])!)
    .attr('stroke', 'white')
    .style('stroke-width', '1px')
    .on('mouseover', function (e, d) {
      d3.select(this).transition().duration(200).attr('transform', 'scale(1.03)');
      hoverText.text(`Author A${d.data[0]}: ${d.data[1]} entries`);
    })
    .on('mouseout', function () {
      d3.select(this).transition().duration(200).attr('transform', 'scale(1)');
      hoverText.text('');
    });

  g.selectAll('text')
    .data(pie)
    .enter()
    .append('text')
    .attr('transform', d => `translate(${arc.centroid(d)})`)
    .attr('dy', '0.35em')
    .style('font-size', '10px')
    .style('text-anchor', 'middle')
    .style('fill', '#fff')
    .text(d => `A${d.data[0]}`);
}


  function drawHighLevelBarChart() {
    const container = d3.select('#highlevel-bar');
    container.selectAll('*').remove();

    const counts = d3.rollup(data, v => v.length, d => d["high-level"]);
    const entries = Array.from(counts, ([label, count]) => ({ label, count }));

    const width = 500, height = 300, margin = { top: 30, right: 20, bottom: 50, left: 60 };
    const svg = container.append('svg').attr('width', width).attr('height', height);

    const x = d3.scaleBand().domain(entries.map(d => d.label)).range([margin.left, width - margin.right]).padding(0.2);
    const y = d3.scaleLinear().domain([0, d3.max(entries, d => d.count)!]).nice().range([height - margin.bottom, margin.top]);

    svg.selectAll('rect')
      .data(entries)
      .enter()
      .append('rect')
      .attr('x', d => x(d.label)!)
      .attr('y', d => y(d.count))
      .attr('width', x.bandwidth())
      .attr('height', d => y(0) - y(d.count))
      .attr('fill', '#8e44ad');

    svg.append('g').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x));
    svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(y));
  }
</script>

<div class="detail-panel">
  <h3>📊 Dataset Details</h3>
  <div class="grid">
    <div class="mini-chart"><h4>Label Distribution</h4><div id="label-bar"></div></div>
    <div class="mini-chart"><h4>Label Over Time</h4><div id="time-line"></div></div>
    <div class="mini-chart"><h4>Author Breakdown</h4><div id="author-pie"></div></div>
    <div class="mini-chart"><h4>High-Level Label Stats</h4><div id="highlevel-bar"></div></div>
  </div>
</div>

<style>
  .detail-panel {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #f8fafc;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 1.5rem;
  }

  .mini-chart {
    background: #fff;
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    position: relative;
  }

  .tooltip {
    position: absolute;
    text-align: center;
    padding: 6px;
    font-size: 0.75rem;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    pointer-events: none;
  }

  h3 {
    margin-bottom: 1rem;
  }

  h4 {
    margin-bottom: 0.5rem;
    color: #2c3e50;
    font-size: 1.1rem;
  }
</style>
