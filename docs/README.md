# docs

Assets referenced by the top-level [README](../README.md).

## demo.gif — not yet recorded

The main README has a commented-out `![Demo](docs/demo.gif)` near the top,
waiting on this file. It is the single highest-impact thing left to add: a
reader decides whether to keep scrolling long before they reach the
architecture section.

### What to capture

About 15 seconds, no narration, no title cards:

1. Upload [`examples/sales_2025.csv`](../examples/sales_2025.csv) — the row and
   column counts appear in the sidebar.
2. Ask *"What are the top 5 products by revenue?"* — let the answer stream in.
3. Expand **How this was calculated** so the generated pandas is visible.
4. Let the chart render.

Step 3 is the one that matters. Every "chat with your CSV" demo looks identical
until the moment the viewer sees real code producing the number — that is the
whole argument of this project, and it is far more convincing shown than
described.

### Recording notes

- 1280×720 or wider; the sidebar has to stay readable.
- Keep it under about 5 MB or GitHub will be slow to render it inline.
- Light theme records better against GitHub's default page.
- Use a real API key, then rotate it — the sidebar shows a masked field, but
  anything in the browser during recording is in the file.

Once recorded, drop it here as `demo.gif` and uncomment the image line in the
main README.
