# Repo

pnpm + Nx monorepo.

- `packages/webgllm`: the engine, LLM inference on WebGL 2 (npm `webgllm`). Its bench pages are in `packages/webgllm/bench`.
- `apps/web`: the docs site (Fumadocs on TanStack Start), prerendered static, deployed to Cloudflare.

Commands:

- `pnpm install`
- `pnpm nx run-many -t build typecheck`
- `pnpm --dir packages/webgllm test`

Cross-runtime benchmarks (webgllm against Transformers.js and wllama) and the TanStack AI adapter live in tanstack-ai-browser, not here.

# Rules

- No per-model tuning tables, no user-agent or device sniffing. Every knob follows one general rule (measured at runtime where it must differ).
- Test only regular ONNX and GGUF builds (e.g. `onnx-community/LFM3-1.7B-ONNX`, `LiquidAI/LFM3-1.7B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`), never Web builds.
- Isolate and prove a bug's cause before fixing it, and discuss it first. No guess fixes.
- Never open, read, print or grep a `.env` file. Read `.env.example` for variable names; ask the user about values.
- No AI attribution in commits or PRs: no `Co-Authored-By: Claude`, no "Generated with", no mention of Claude or Anthropic.

# Benchmarks and devices

Test on the Mac unless the user asks for a device. Another agent may be driving the Pixel 10 Pro: use it only when the user asks, and follow "Controlling the Pixel" below (port 9222 is the Pixel's Chrome, forwarded by adb).

On the Mac, test in the desktop app's built-in Browser (the `mcp__Claude_Browser__*` tools), never in the user's own browser: no Claude in Chrome, no Safari or Chrome windows of theirs.

## Running the engine benches

The pages live in `packages/webgllm/bench` and are served cross-origin isolated by `server.mjs`, which also bundles the package's TypeScript on request (`/bundle/<path>.js` is `packages/webgllm/<path>.ts`). `bench/README.md` has every page and parameter.

1. Start the server with absolute paths, on a free port (the docs site's dev server uses 3000, its static preview 8788):
   `HF_DIR=/path/to/lfm3-tokenizer/ MODELS_DIR=/path/to/models/ node /Users/david/Developer/webgllm/packages/webgllm/bench/server.mjs 8790`
   - `HF_DIR`: `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` from https://huggingface.co/LiquidAI/LFM3-1.7B.
   - `MODELS_DIR`: GGUFs, and the phone driver's `job.json` (below).
   - `REF_DIR`, `REF_ONNX_DIR`: llama.cpp and onnxruntime reference tensors for `check.html`. `ONNX_COMMUNITY_DIR`, `ONNX_DIR`: ONNX builds for it.
   - `RESULTS_FILE`: where `POST /result` lines go.
2. Open pages in ONE tab of your own that you reuse for every run. Never close or navigate the user's other tabs.
   - `webgl-perf.html?id=<run>&production=1&rounds=3&len=512`: load, prompt and writing throughput with painted frames during every phase (FPS while the canvas repaints). Check the painted-frame numbers, not only tokens/s: the engine must not freeze the page.
   - `check.html`: correctness against the reference tensors, layer by layer; `?smoke=1&len=512` reads and answers a chat prompt with no reference and ends with `LOGITS HASH`.
   - `mem.html?model=qwen-gguf&len=2048&follow=1024&reply=128&target=8192&ctx=8192`: memory and speed over a chat (`model=qwen-onnx|qwen-gguf|lfm-onnx|lfm-gguf` or a full name).
   - `gl-info.html`: what the browser's WebGL 2 offers.
3. Read results in the server's output: each page line is printed as `[page] ...`. `ALL DONE` ends a page; `ERROR` is a failure.
   - A loop that waits for `ALL DONE` must also stop on `ERROR`, an abort and a dead server.
4. Keep the port the same between runs: a page's origin holds its downloaded models (OPFS), and a new port downloads them again.
5. Measure memory:
   - `mem.html`'s texture log (live and peak textures by category).
   - macOS `footprint -p <pid>` of the desktop app's `Claude Helper` renderer and GPU processes.
   - On a Mac, use the iPhone's `ctx=8192`, or the cache is sized for the model's whole window.
6. Comparing builds or settings: alternate them run by run (A, B, A, B) in one page and session. Fix the weight layout (`tune=layerRgba4:1`) when comparing on an M3.

## Controlling the iPhone

The user's iPhone 16 Pro Max is "Dave iPhone", UDID `00008140-000A7D0E227B001C`. Only use it when the user asks, and say what you're doing on it.

**Tabs:** open only ONE tab of your own on the phone, and do all your work in it (the driver page runs every job in its frame). Never close, reload or navigate the user's other tabs.

- **HTTPS:** WebGL storage (OPFS) needs HTTPS on the phone, so tunnel the bench server. The user has approved either:
  - a Cloudflare tunnel: `cloudflared tunnel --url http://localhost:8790`, which prints an `https://….trycloudflare.com` address;
  - ngrok: `ngrok http 8790`.
  - Don't stop or change the user's own ngrok tunnels.
- **Benchmarks:** open `https://<tunnel>/driver.html` on the phone once. It polls `$MODELS_DIR/job.json` and loads each new job in a frame.
  - Run a job: write `{"id":"<unique>","url":"/mem.html?model=qwen-gguf&ctx=8192"}` (or a `webgl-perf.html` URL) to that file.
  - Park the phone afterwards with `"url":"about:blank"`.
  - Results arrive in the server's output as on the Mac.
  - A second `Mozilla/5.0 (iPhone` line means the page reloaded: Safari crashed and recovered.
  - Rest the phone a minute or two between jobs; it heats up.
- **Driving Safari (USB):** Safari's Web Inspector must be on (Settings > Apps > Safari > Advanced).
  - `ios_webkit_debug_proxy -c 00008140-000A7D0E227B001C:9333 --no-frontend`
  - `curl -s http://127.0.0.1:9333/json` lists the tabs.
  - `node packages/webgllm/bench/ios-tab.mjs <webSocketDebuggerUrl> '<async function body>'` runs JavaScript in a tab and prints its console.
- **Memory kills (USB):**
  - `idevicesyslog -u 00008140-000A7D0E227B001C --no-colors | grep -aE "per-process-limit|processTerminated|EXC_RESOURCE|vm-compressor"`
  - iOS kills a Safari tab near 2 GB (`per-process-limit`) and writes no crash report for it, so this live log is the only record.
- **Over Wi-Fi:** only crash reports, via `xcrun devicectl device info files --device 00008140-000A7D0E227B001C --domain-type systemCrashLogs`. libimobiledevice and the Web Inspector proxy need USB.
- **Behaviours to expect:**
  - iOS suspends a page whose screen is off or whose tab is in the background, which stops downloads and closes OPFS access handles. Ask the user to keep Safari in front and tap the page if it stalls.
  - Safari also deletes a site's storage once per Safari launch (`WebsiteDataStore::removeData`).
  - Load models with an 8,192-token context on iOS.
- **iOS Simulator:** a stand-in for the phone, with limits.
  - No 2 GB limit, and its GPU is the Mac's.
  - Measure its page with `footprint` on the `CoreSimulator` `com.apple.WebKit.WebContent` process.
  - Driving it with the simulator tool: take a screenshot after every action. The first tap after a page change is often lost (tap again), and typed text can arrive late.

## Controlling the Pixel

The user's Pixel 10 Pro (Tensor G5, Imagination PowerVR GPU) is adb serial `57171FDCH00AT3`; adb is `~/Library/Android/sdk/platform-tools/adb`. Only use it when the user asks, and say what you're doing on it.

**Sharing:** other sessions often drive the phone at the same time.
- Before starting, look at what's already there:
  - `adb -s 57171FDCH00AT3 reverse --list` and `adb forward --list`;
  - `curl -s localhost:9222/json/list` (Chrome's open tabs).
- Add your own port mappings; never remove or change another session's.
- Open ONE tab of your own and do all your work in it. Never close, reload or navigate other tabs.
- If another session is running a benchmark, wait for it to finish instead of competing: the results of both would be wrong.

**Connecting (USB):**
- `adb -s 57171FDCH00AT3 reverse tcp:8790 tcp:8790` makes the Mac's bench server `http://localhost:8790` on the phone. localhost is a secure context, so storage works without a tunnel. Keep the same port between runs, as on the Mac (OPFS).
- `adb -s 57171FDCH00AT3 forward tcp:9222 localabstract:chrome_devtools_remote` exposes Chrome's DevTools protocol on the Mac's port 9222.
  - `curl -s localhost:9222/json/list` lists the tabs with their `webSocketDebuggerUrl`.
  - Drive a tab over that WebSocket (`Page.navigate`, `Runtime.evaluate`, console events), or with playwright-core's `chromium.connectOverCDP("http://localhost:9222")`. playwright-core is not a repo dependency, so install it in a scratch folder.
- Open a page from the shell: `adb -s 57171FDCH00AT3 shell am start -a android.intent.action.VIEW -d '<url>' com.android.chrome`.
- **Benchmarks:** the driver page works as on the iPhone. Open `http://localhost:8790/driver.html` once, then write jobs to `$MODELS_DIR/job.json`, and park the phone with `"url":"about:blank"` afterwards.

**Idle check:** run it before and after every measurement. Discard a run if the phone wasn't idle, or if another tab appeared or changed during it.
- GPU clock: `adb -s 57171FDCH00AT3 shell cat /sys/class/devfreq/34f00000.gpu0/cur_freq`. It reads `396000000` when idle; anything higher means something is using the GPU.
- Thermal: `adb -s 57171FDCH00AT3 shell dumpsys thermalservice | grep "Thermal Status"`. 0 is cool and 1 is light. Above 1, rest the phone before measuring.
- Busy renderers: `adb -s 57171FDCH00AT3 shell top -b -n 1 -m 12 | grep -E "chrome|sandbox|privileged"`.

**Heat changes the results:**
- The CPU throttles hard as the phone warms (CPU runtimes lose about half their speed); the GPU holds steadier, but rest it anyway.
- Compare builds only within one session: alternate them run by run (A, B, A, B) and record the thermal status with each result. Never compare against numbers from another day or temperature.
- Rest the phone a minute or two between long runs.

**PowerVR notes for this engine:**
- The engine picks byte texels over RGBA4 there by timing at load (`Engine.chooseLayout`), not by GPU name.
- PowerVR merges and skips full-screen draws whose output is overwritten later, so a timed draw must read the previous draw's output (`bench/README.md`).
- A readback costs 10 to 35 ms of fixed latency; `unpackUnorm2x16` returns wrong values on that driver.

**Crashes and memory:**
- `adb -s 57171FDCH00AT3 logcat -d | grep -iE "lowmemorykiller|lmkd|chromium.*crash"` shows tabs Android killed.
- `adb -s 57171FDCH00AT3 shell dumpsys meminfo com.android.chrome` gives Chrome's memory.
