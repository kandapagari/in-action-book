// Wraps every <table> in <div class="table-wrap"> so wide tables can break
// out of the narrow prose measure into the full content width and fall back
// to horizontal scroll when they exceed it. Applied site-wide via the
// markdown rehype pipeline (see astro.config.mjs), so every book table —
// chapters and appendices — gets the same treatment.

export default function rehypeTableWrap() {
  return (tree) => wrapTables(tree);
}

function wrapTables(node) {
  const children = node.children;
  if (!children) return;
  for (let i = 0; i < children.length; i++) {
    const child = children[i];
    if (child.type === 'element' && child.tagName === 'table') {
      children[i] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-wrap'] },
        children: [child],
      };
      wrapTables(child);
    } else {
      wrapTables(child);
    }
  }
}
