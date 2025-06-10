import * as d3 from 'd3';

interface LabelInfo {
  short: string;
  color: string;
}

export function generateLabelMap(data: { label: string }[]): Record<string, LabelInfo> {
  const labels = Array.from(new Set(data.map(d => d.label).filter(Boolean)));
  const usedShorts = new Set<string>();
  const labelMap: Record<string, LabelInfo> = {};
  const colorScale = d3.scaleOrdinal<string, string>(d3.schemeCategory10).domain(labels);

  function generateUniqueShort(label: string): string {
    const words = label.split(/[\s\-]/).filter(Boolean);
    let short = words.map(w => w[0].toUpperCase()).join('');

    if (!usedShorts.has(short)) {
      usedShorts.add(short);
      return short;
    }

    for (let len = 2; len <= label.length; len++) {
      const fallback = label.replace(/[^A-Za-z]/g, '').substring(0, len).toUpperCase();
      if (!usedShorts.has(fallback)) {
        usedShorts.add(fallback);
        return fallback;
      }
    }

    const fallback = label.toUpperCase();
    usedShorts.add(fallback);
    return fallback;
  }

  for (const label of labels) {
    const short = generateUniqueShort(label);
    labelMap[label] = {
      short,
      color: colorScale(label),
    };
  }

  return labelMap;
}