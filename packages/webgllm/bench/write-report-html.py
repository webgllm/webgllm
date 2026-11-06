"""Render selected benchmark summaries as a local, script-free comparison page."""
import html
import json
import math
import pathlib
import sys

output, markdown, *summaries = sys.argv[1:]
data = [json.loads(pathlib.Path(p).read_text()) for p in summaries]
esc = lambda value: html.escape(str(value))
def number(value, places=1):
    return f'{value:.{places}f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'not captured'
def pair(before, after, scale=1, places=1):
    return f'<span class="before">{number(before / scale if before is not None else None, places)}</span> &rarr; <strong>{number(after / scale if after is not None else None, places)}</strong>'
def table(headings, rows):
    return '<div class="table"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headings) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'
performance, startup, memory = [], [], []
for item in data:
    device = esc(item['device'])
    for name, suffix in [('Short', 'Short'), ('Long', 'Long')]:
        before, after = item['runs']['before' + suffix], item['runs']['after' + suffix]
        bm, am = (item['memory']['before'], item['memory']['after']) if suffix == 'Short' else (item['memory']['beforeLong'], item['memory']['afterLong'])
        performance.append([device, f"{name} &middot; {after['inputTokens'][0]} tokens", pair(before['prefillFps'], after['prefillFps']) + '<br><small>generation: ' + pair(before['decodeFps'], after['decodeFps']) + '</small>', pair(before['aggregatePromptTps'], after['aggregatePromptTps']), pair(before['aggregateDecodeTps'], after['aggregateDecodeTps'], places=2), pair(bm['peakGpuBytes'], am['peakGpuBytes'], 2**20)])
        kind = bm['kind']
        key, scope = ('peakWorkerBytes', 'Worker V8 + backing + embedder peak') if kind == 'v8-worker' else ('peakPageBytes', 'WebKit process resource categories peak') if kind == 'webkit' else ('retainedPageBytes', 'Page + dedicated worker retained after GC')
        memory.append([device, name, esc(scope), pair(bm.get(key), am.get(key), 2**20)])
    before, after = item['load']['before'], item['load']['after']
    startup.append([device, pair(before['cold']['medianMs'], after['cold']['medianMs'], 1000, 2), pair(before['cached']['medianMs'], after['cached']['medianMs'], 1000, 2), pair(before['cold'].get('fullyCalibratedMedianMs'), after['cold'].get('fullyCalibratedMedianMs'), 1000, 2)])
pixel = next(x for x in data if x['device'].startswith('Pixel'))
maximum_load = max(pixel['load']['after'][x]['maxMs'] for x in ['cold', 'cached']) / 1000
worst_gap = pixel['load']['after']['cold']['worstFrameMs']
motion_html = '<details><summary>Recorded Pixel prefill motion: before / after</summary><p>Separate screen-recording diagnostic, with encoder overhead. The clips show only the animated canvas.</p><video controls loop muted playsinline preload="none" width="600" src="results/presentation-before.mp4"></video><video controls loop muted playsinline preload="none" width="600" src="results/presentation-after.mp4"></video></details>'
body = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WebGL performance report</title>
<style>
:root {{ color-scheme: dark; font: 15px/1.5 system-ui, sans-serif; background: #13171c; color: #eef3f8; }}
body {{ max-width: 1240px; margin: 28px auto; padding: 0 24px 40px; }}
h1 {{ font-size: 30px; margin: 4px 0; letter-spacing: -.8px; }}
h2 {{ font-size: 19px; margin: 25px 0 9px; }}
p {{ margin: 9px 0 15px; }}
a {{ color: #8ceac4; }}
strong {{ color: #eef3f8; }}
.eyebrow, .before, small {{ color: #aebbc9; }}
.eyebrow {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }}
.notice {{ border-left: 3px solid #8ceac4; background: #1c252c; padding: 12px 16px; margin: 18px 0; }}
.table {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #34414c; vertical-align: top; }}
th {{ color: #bccbd8; font-size: 12px; font-weight: 600; }}
td {{ white-space: nowrap; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; color: #c7d2dd; }}
summary {{ cursor: pointer; padding: 12px 0; }}
video {{ max-width: 100%; height: auto; }}
</style>
<div class="eyebrow">LFM3 1.7B &middot; WebGL 2 &middot; perf/webgl-performance</div>
<h1>WebGL performance report</h1>
<p>Before &rarr; after. Models released; this report runs no model worker.</p>
<div class="notice"><strong>Pixel targets passed in the measured trials.</strong> All final cached-file loads were under 10 seconds (maximum {maximum_load:.2f}s). Cold startup still had a {worst_gap:.1f}ms worst frame gap.<br>
<strong>iPhone version coverage:</strong> iPhone figures use checkpoint <code>70d3ea0</code>. Later startup and fast-GPU refinements were tested on Pixel and Mac; they still need an iPhone recheck.</div>
<h2>Throughput, animation and GPU memory</h2>
<p class="eyebrow">Overall token rates &middot; FPS estimated over each phase from requestAnimationFrame &middot; GPU allocation MiB</p>
{table(['Device', 'Prompt', 'Prefill FPS', 'Prompt tok/s', 'Generated tok/s', 'GPU peak MiB'], performance)}
<h2>Loading the cached model file</h2>
<p>Median seconds. First download of the 1.05 GB GGUF is excluded. The original tuned in the background; the updated worker finishes tuning before ready.</p>
{table(['Device', 'Fresh calibration: first ready', 'Saved calibration: ready', 'Fresh calibration: fully tuned'], startup)}
<h2>CPU memory counters</h2>
<p>MiB. Scopes differ across browsers; compare within one device. GPU counters exclude driver overhead. These counters are not whole-process RSS.</p>
{table(['Device', 'Prompt', 'Counter scope', 'Before → after, MiB'], memory)}
<p>iPhone WebKit resource peaks do not establish a CPU-memory reduction; retained loaded-minus-unloaded categories increased about 2–3 MiB, while GPU allocations decreased about 15 MiB.</p>
<p><a href="{esc(pathlib.Path(markdown).name)}" download>Detailed Markdown report</a> &middot; <a href="results/source-manifest.json">Source and build provenance</a></p>
{motion_html}
<details><summary>Full report: trials, ranges, frame gaps, quality and limitations</summary><pre>{esc(pathlib.Path(markdown).read_text())}</pre></details>
</html>'''
pathlib.Path(output).write_text(body)
