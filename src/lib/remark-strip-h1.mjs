// Removes the first H1 heading from a markdown AST. The book's section
// files open with `# 1.1 Title`; the section page renders the title from
// frontmatter, so the body H1 would be a duplicate.

export default function remarkStripH1() {
  return (tree) => {
    const idx = tree.children.findIndex(
      (node) => node.type === 'heading' && node.depth === 1,
    );
    if (idx >= 0) tree.children.splice(idx, 1);
  };
}
