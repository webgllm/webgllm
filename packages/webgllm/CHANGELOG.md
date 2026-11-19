## 0.3.8 (2026-11-19)

### 🚀 Features

- webgpu option: "auto" or false keeps models on WebGL 2 ([8c93eb5](https://github.com/webgllm/webgllm/commit/8c93eb5))
- **engine:** WebGPU by default where the browser has it, WebGL 2 elsewhere (TUNING.webgpu 1): with the defaults, LFM3 1.7B reads prompts at 5,175 tokens/s on an M3 Ultra (Chrome), 774 on an iPad Pro M4 (Safari), 187-200 on a Pixel 10 Pro (Chrome), and Firefox for Android, which has no WebGPU, keeps WebGL at 82-84 ([f47c19c](https://github.com/webgllm/webgllm/commit/f47c19c))
- **engine:** TUNING.fusedAdd 2 is a prompt's add too, 1 a token's alone (for A/Bs of the prompt's on other GPUs) ([15bb7b7](https://github.com/webgllm/webgllm/commit/15bb7b7))
- **engine:** TUNING.glColumnBands paces WebGL's prompt matmuls in bands of columns (off; for A/Bs); column bands only for passes of 8 rows or more ([d9b8af2](https://github.com/webgllm/webgllm/commit/d9b8af2))
- **webgpu:** shared K loops where they pay (tuned), prompt passes in 8×8 workgroups, WebGL when WebGPU fails ([b69c547](https://github.com/webgllm/webgllm/commit/b69c547))
- **webgpu:** the engine on WebGPU, its WebGL kernels translated to compute shaders (TUNING.webgpu, off by default) ([54ace3d](https://github.com/webgllm/webgllm/commit/54ace3d))
- **webgllm:** predictSpeed, a model's prompt and writing speed on this device before downloading it ([f89d204](https://github.com/webgllm/webgllm/commit/f89d204))

### 🩹 Fixes

- **pool:** an eviction finds a model an unload already removed and waits for its memory too ([f0ef6b6](https://github.com/webgllm/webgllm/commit/f0ef6b6))
- **pool:** a kept worker is never an ended or failed one, and an eviction completes once the evicted model is freed ([9b3ec3f](https://github.com/webgllm/webgllm/commit/9b3ec3f))
- **loaders:** retired read buffers leave the ONNX loader's pieces whole where they aren't reused ([b3d0b46](https://github.com/webgllm/webgllm/commit/b3d0b46))
- **pool:** a load out of room canceled and unloaded by its failure's listener leaves no unhandled rejection ([3cd058d](https://github.com/webgllm/webgllm/commit/3cd058d))
- **webgpu:** deleting a texture a queued pass reads leaves no bind group for it ([13f39c4](https://github.com/webgllm/webgllm/commit/13f39c4))
- **pool:** a worker whose model was unloaded is kept for the next load, for 30 s ([d9b7ae1](https://github.com/webgllm/webgllm/commit/d9b7ae1))
- **webgpu:** unloading frees the model's GPU memory: dispose destroys every texture and buffer at once ([18b1b9d](https://github.com/webgllm/webgllm/commit/18b1b9d))
- **loaders:** pieces reuse earlier pieces' buffers only when the consumer opts in (reusePieces) ([9b6e687](https://github.com/webgllm/webgllm/commit/9b6e687))
- **api:** unloading ends a load's storage-prompt wait, a caller's worker is told the webgpu setting on every attachment, and speed predictions abort per call ([efdb712](https://github.com/webgllm/webgllm/commit/efdb712))
- **webgpu:** queued passes keep their order, throttled pages keep backpressure, cooperating lanes add the starting value once, stored 8-bit prompts take the native fallback, and the fingerprint covers the backend ([8b829c0](https://github.com/webgllm/webgllm/commit/8b829c0))
- **pool:** a canceled load stops waiting for the storage prompt, and a retried download frees its failed attempt's worker ([32ea2e4](https://github.com/webgllm/webgllm/commit/32ea2e4))
- **engine:** chooseHead times 12 heads at a time where the split Q6_K head competes, and more rounds while the fastest two are close ([19fcf88](https://github.com/webgllm/webgllm/commit/19fcf88))
- **engine:** attention scores stored as halves whatever the embedding, not only beside a compact Q6_K head ([e6068c3](https://github.com/webgllm/webgllm/commit/e6068c3))
- **engine:** the packing probe still times pack 3 alone where stored 8-bit matrices allow nothing else ([b0ad6a6](https://github.com/webgllm/webgllm/commit/b0ad6a6))
- **engine:** stored 8-bit matrices keep the h2 weight scratch in every prompt packing ([95087e8](https://github.com/webgllm/webgllm/commit/95087e8))
- **engine:** an ONNX Qwen3.5's 4-bit beta|alpha in a storage buffer takes the scratch path in prompts ([467b35f](https://github.com/webgllm/webgllm/commit/467b35f))
- **predict-speed:** measures on the backend a model's worker runs on (WebGPU where the browser has it, else WebGL), and keys its remembered predictions by backend: it had always measured WebGL. LFM3 1.7B predicted at 5,002 tokens/s prompt and 239 written on an M3 Ultra (measured 5,175 and 245), 780 and 78 on an iPad Pro M4 (774 and 82), 100 and 9.5 in Firefox on a Pixel 10 Pro (82-84 and 9.1-9.6); bench: predict-speed.html?webgpu=0|1 ([682e088](https://github.com/webgllm/webgllm/commit/682e088))
- **paced:** column bands' least width is a column quantum, not a band's floor of texels: 240-column bands left an iPhone 16 Pro Max's slices half-filled at each seam and read Qwen3.5 2B's prompts at 323-358 tokens/s where 347-363 (row bands 349-366), LFM2.5 1.2B's at 468-496 where 474-495; an M3 Ultra the same ([4f28150](https://github.com/webgllm/webgllm/commit/4f28150))
- **engine:** the packing probe holds only the timed pack's f16 scratch (LFM3's load peak on an iPhone 16 Pro Max 1,263 MB where 1,323); the schema constraint keeps tokenizer.json's decoder, vocabulary and added tokens, not the whole of it (with huggingface/tokenizers.js#54, LFM3's tokenizer heap 41 MiB where 73) ([#54](https://github.com/webgllm/webgllm/issues/54))
- **engine:** where the lanes win, tune times bufferMatvec against bufferPairs (tuned.matvec) and the lanes' trial runs bufferMatvec with them, so a GPU whose lanes win by noise keeps bufferPairs; bufferMatvec splits nibbles with unpack4x8unorm ([9e81f26](https://github.com/webgllm/webgllm/commit/9e81f26))
- **engine:** the warm-up's prompt batch is a full batch, so a cached load's first prompt compiles nothing (its passes' WebGPU variants depend on their size) ([21c921c](https://github.com/webgllm/webgllm/commit/21c921c))
- **engine:** the packing probe drops a cold first round, so a faster layout can replace a slower one this GPU chose before ([6d6c74e](https://github.com/webgllm/webgllm/commit/6d6c74e))
- **webgpu:** pace overrides the paced GPU's, so the library builds again ([a1fd713](https://github.com/webgllm/webgllm/commit/a1fd713))
- a request that leaves out the context length runs at the one its model is loaded at ([8d4944e](https://github.com/webgllm/webgllm/commit/8d4944e))
- **tuning:** the head probe times the heads' GPU work; norms after a fused add read its one sums column ([1f98573](https://github.com/webgllm/webgllm/commit/1f98573))
- **webgpu:** the head probe times its candidates without panels where the embedding has none ([bff8e0a](https://github.com/webgllm/webgllm/commit/bff8e0a))
- **tuning:** a choice that kept TUNING's value stands like any other; a 5% win in every round replaces one ([56dd876](https://github.com/webgllm/webgllm/commit/56dd876))
- **tuning:** a GPU's kernel choices stand unless another way is clearly faster ([6318a0c](https://github.com/webgllm/webgllm/commit/6318a0c))
- **memory:** Qwen3.5 2B no longer crashes iPhone Safari after loading ([8954700](https://github.com/webgllm/webgllm/commit/8954700))
- fused f16 prompt inputs only where every consumer reads them; free upload temporaries ([e25c06b](https://github.com/webgllm/webgllm/commit/e25c06b))
- Firefox for Android stores models; Qwen3.5 2B loads on 8,192-texel GPUs; the self-check covers the head ([089e46b](https://github.com/webgllm/webgllm/commit/089e46b))
- **bench:** only LFM's reordered layers are flagged (Qwen's keep their layout); the generator relinks weights an earlier run left ([3837104](https://github.com/webgllm/webgllm/commit/3837104))
- **bench:** check.html reorders a reference's q/k heads only where the loader did; an onnxruntime reference generator ([a8f9eec](https://github.com/webgllm/webgllm/commit/a8f9eec))

### 🔥 Performance

- **webgl:** a split Q6_K embedding uploaded a loader's chunk at a time, not 128 row groups ([0802629](https://github.com/webgllm/webgllm/commit/0802629))
- **webgl:** a Q6_K embedding's head split into its weights' low 4 bits (RGBA4) and top 2, where it times faster ([6cf1ed3](https://github.com/webgllm/webgllm/commit/6cf1ed3))
- **load:** a GGUF's Q5_K and Q4_1 tensors converted to 8 bits through WebAssembly SIMD ([88e6450](https://github.com/webgllm/webgllm/commit/88e6450))
- Qwen3.5 2B GGUF closer to its ONNX build: 8-bit prompts through the f16 scratch on WebGL, a cheaper Q6_K head, 8-bit norms folded ([a96486e](https://github.com/webgllm/webgllm/commit/a96486e))
- **load:** raw MatMulNBits rows repacked through WebAssembly SIMD, not a GPU pass ([f1d7259](https://github.com/webgllm/webgllm/commit/f1d7259))
- **webgpu:** 4-bit embeddings in word texels with a native head ([8529ce6](https://github.com/webgllm/webgllm/commit/8529ce6))
- **engine:** ONNX models as fast as GGUF: the WebGPU buffer kernels (bufferMatvec, bufferPairs, bufferDequant) read zero points packed with the scales, so an ONNX model's asymmetric matrices go into storage buffers too (and the buffer GEMM takes its prompts); raw MatMulNBits rows go into byte texels through a GPU pass (K.rawQ4Bytes, TUNING.rawUploadGpu) instead of being repacked on the CPU. LFM3 1.7B ONNX on an M3 Ultra: WebGPU 5,150 prompt tokens/s and 236 written where 4,650 and 216 (GGUF 5,150 and 241), loads in 670 ms where 940 (GGUF 620); WebGL loads in 770 ms where 970 (GGUF 650), prompts 3,200 where 3,030 (GGUF 3,020-3,220); replies unchanged. The upload staging textures are freed after a load. bench: matrix layout flags and load-step times after the warm-up ([f8a221f](https://github.com/webgllm/webgllm/commit/f8a221f))
- **predict-speed:** its prompts run every row through every layer (the last layer's last-row tail and its programs left out: 39 programs where 47, an iPhone 16 Pro Max's guess 365 tokens/s where measured 360-385), and a model's shapes are remembered (webgl-models/speed-shapes.json): a later measurement skips the GGUF header's download, 2.4 s to 5 ms on the iPhone; bench: predict-speed.html?tag= for devices at once ([6db0f5a](https://github.com/webgllm/webgllm/commit/6db0f5a))
- **predict-speed:** a fast guess: predictSpeed(model, { decode: false }) times the prompt alone, LFM3 1.7B in 1.7-1.9 s in Firefox on a Pixel 10 Pro (14 s before), 1.3-1.5 s in its Chrome, 0.3-0.9 s on an iPad Pro M4, 0.2 s on an M3 Ultra; a 256-token prompt (the prompt's own kernels, TUNING.quadMinBatch) through one stack, compiled through a 64-token run of the first and last layers; the load's packing, head and (prompt-only) matvec choices skipped; an 8,192-row head; one timed run, not rounds. With writing: whole-model forwards and the head at two sizes, 2.7 s in the Pixel's Firefox ([890deb6](https://github.com/webgllm/webgllm/commit/890deb6))
- **engine:** a prompt's fused add, its queries' prep and its short convolution as single native passes only where tune's lanes won (TUNING.fusedAdd 3, promptPrep and promptConv -1; see workgroupFusions): a Pixel 10 Pro, whose lanes lose, reads LFM2.5 350M's prompts at 912-917 tokens/s this way where 789-799 fused, 1.2B's at 293-302 where 280-282, LFM3's and Qwen3.5 2B's the same or better; an M3 Ultra, whose lanes win, keeps them ([6b07e9a](https://github.com/webgllm/webgllm/commit/6b07e9a))
- **gl:** a prompt's wait doesn't sleep on timers where its fence polls wait inside the call (Firefox): on a Pixel 10 Pro it reads LFM3's prompts at 96 tokens/s where 92, Qwen3.5 2B's at 86 where 80, LFM2.5 350M's at 448 where 435 and 1.2B's at 144 where 142, the same frames ([17a9a38](https://github.com/webgllm/webgllm/commit/17a9a38))
- **webgpu:** a prompt batch's short convolution and its state are one native pass (TUNING.promptConv, gpu/native.ts's promptConv): a workgroup an h2 texel of the output and a block of tokens, B·x once a token through workgroup memory, where two passes multiplied it again for each tap of every output texel. The same bits (taps in order, each a fused multiply-add): logprobs agree to 6 decimals over 256 tokens on the three LFM models. An M3 Ultra runs LFM3's (7 taps) in 0.07 ms a layer where 0.20, and reads its prompts at 5,410 tokens/s where 5,330, LFM2.5 1.2B's at 7,750 where 7,690, 350M's at 22,030 where 21,870; TUNING.bufferGemm8's note: the pair matmul ran at 2.7 TFLOP/s ([b5ee58d](https://github.com/webgllm/webgllm/commit/b5ee58d))
- **webgpu:** a prompt batch's queries, keys and values are one native pass (TUNING.promptPrep, gpu/native.ts's promptPrep): a workgroup a head of a token, the head's sum of squares through workgroup memory, where three passes each summed the head's squares again for every texel. The same bits (the sum as the GLSL pass's compiles: a fused multiply-add a texel): logprobs agree to 6 decimals over 256 tokens on all four models. Not for a batch that wraps around its ring, nor image rows. An M3 Ultra runs a layer's in 0.04 ms where 0.39, and reads LFM3 1.7B's prompts at 5,300 tokens/s where 5,155, Qwen3.5 2B's at 4,880 where 4,765, LFM2.5 1.2B's at 7,680 where 7,610, 350M's the same ([73b07a6](https://github.com/webgllm/webgllm/commit/73b07a6))
- **webgpu:** a prompt batch's recurrence chunks run one after another in one native dispatch (TUNING.chunkLoop, gpu/native.ts's linChunkLoop): a workgroup a value texel, its column of the state in workgroup memory from chunk to chunk, where each chunk was two dependent passes; segments end at a row whose state is saved, and a chunk a saved prefix ends inside keeps the two passes. The same sums in the same order: Qwen3.5 2B's logprobs are the same to 6 decimals over 256 tokens. An M3 Ultra runs a layer's 17 chunks of a 531-token prompt in 0.38 ms where 0.75, and reads Qwen3.5 2B's prompts at 4,590 tokens/s where 4,440 ([12ff067](https://github.com/webgllm/webgllm/commit/12ff067))
- **webgpu:** Gated DeltaNet's 8-bit beta|decay matmul goes through the scratch buffer too, into the block's output textures (free until out_proj, and a 4-token matmul's pair): its 32 rows over all of K took 0.25 ms a layer as a pair matmul on an M3 Ultra and the gates' pass after it 0.18, 0.05 and 0.02 dequantized and in 4 K-slices. Qwen3.5 2B's prompts read at 4,530 tokens/s where 4,450, with the same replies ([c803b4f](https://github.com/webgllm/webgllm/commit/c803b4f))
- **webgpu:** 8-bit layer matrices among 4-bit ones go through the f16 scratch buffer and the native prompt matmul too (TUNING.bufferGemm8), where the native pair matmul ran them at 2.7 TFLOP/s (the f16 matmul: 17 to 19): unsloth's Qwen3.5 2B has 21 (its Q5_K ssm_out and Q4_1 ffn_down tensors, kept as 8-bit). Only through the scratch the 4-bit matrices already have, and only a matrix no wider than it was sized for: no memory more. An M3 Ultra reads Qwen3.5 2B's prompts at 4,285 tokens/s where 3,504, with the same replies; LFM models have no such matrices ([5b5e492](https://github.com/webgllm/webgllm/commit/5b5e492))
- **webgpu:** the buffer dequant takes a row's block a thread: its 4 words and its scale fetched once for the block's 4 h2 texels, where each texel fetched two words and the scale twice; the same bits (bench/gemm-exact.html checks them), 152 µs for LFM3's four prompt matrices on an M3 Ultra where 270. It reads LFM2.5 350M's prompts at 22,320 tokens/s where 22,060, 1.2B's at 7,640 where 7,470, LFM3's at 5,180 where 5,090, Qwen3.5 2B's at 3,470 where 3,420. bufferDequant also takes 8-bit weights. bench: gemm-exact times the engine's own matmul and dequant, the dequant a texel, a word pair or a row group's block a thread, and a matmul straight from the 4-bit weights (0.37-0.41x the f16 matmul's speed: no dequant pass doesn't pay for decoding every weight again for each token tile) ([63983ec](https://github.com/webgllm/webgllm/commit/63983ec))
- **webgpu:** a prompt's residual add is one native pass too: fusedAdd reads pair tensors and 4-token matmuls' outputs (a workgroup a row of y, each texel fetched once), where prompts drew y's sums, the add and its sums as three passes; its threads are a power of two for any hidden size. An M3 Ultra reads LFM3 1.7B's prompts at 5,060 tokens/s where 4,920, LFM2.5 350M's at 21,950 where 21,570, 1.2B's at 7,410 where 7,350, Qwen3.5 2B's the same; replies unchanged ([4d072af](https://github.com/webgllm/webgllm/commit/4d072af))
- **engine:** bufferGemm's matmul takes two row groups a thread where tune's lanes lost, and is paced in bands of columns, every token in each: a Pixel 10 Pro reads LFM2.5 350M's prompts at 956-996 tokens/s where 724-797, 1.2B's at 315-320 where 254-258, LFM3's at 198-201 where 164-170, Qwen3.5 2B's at 115-122 where 109-110 ([971435f](https://github.com/webgllm/webgllm/commit/971435f))
- **webgpu:** f16Pack 3's prompt matmuls read the dequantized weights from a storage buffer (native dequant into it, native matmul out of it, a pipeline a slice length); native passes may write storage buffers; bench: pskip_<name> leaves passes out of prompts ([33509bd](https://github.com/webgllm/webgllm/commit/33509bd))
- **webgpu:** 8-bit layer weights go to storage buffers too: the buffer matvecs read them, prompts read them in a native pair matmul, their textures go, and their norms fold ([3c48aea](https://github.com/webgllm/webgllm/commit/3c48aea))
- **webgpu:** 4-bit layer weights go to storage buffers as they load, on every GPU, before any pass binds their textures; tuning keeps bufferPairs; a Gated DeltaNet token's delta, output and state in one native pass ([5ccd216](https://github.com/webgllm/webgllm/commit/5ccd216))
- **webgpu:** prompts' f16 matmuls (f16Pack 0) as a native pass, 8 rows × 4 tokens an invocation with token blocks varying fastest; the packing probe takes it where it's faster ([9061e08](https://github.com/webgllm/webgllm/commit/9061e08))
- **webgpu:** bufferPairs, a buffer matvec writing a slice a lane, for GPUs without shared K loops (off unless buffers are forced); a weight's copy to its buffer is sent at once ([acf7653](https://github.com/webgllm/webgllm/commit/acf7653))
- **webgpu:** where the lanes share K loops, 4-bit byte weights move to storage buffers, read by a native matvec and the prompt dequant ([9a28da2](https://github.com/webgllm/webgllm/commit/9a28da2))
- **webgpu:** native attention's scores two threads a key and its values summed by every thread; shapes past 16 KiB of workgroup memory keep the old passes ([874302c](https://github.com/webgllm/webgllm/commit/874302c))
- **webgpu:** a token's attention as two native passes, its scores in workgroup memory ([203ebb7](https://github.com/webgllm/webgllm/commit/203ebb7))
- **webgpu:** the pick's chunk maxima and argmax as native passes ([5eb0e41](https://github.com/webgllm/webgllm/commit/5eb0e41))
- **webgpu:** draft batches' pipelines compile in the background after the load ([f9341cd](https://github.com/webgllm/webgllm/commit/f9341cd))
- **webgpu:** the warm-up compiles its pipelines in the background, several at once ([dd6149f](https://github.com/webgllm/webgllm/commit/dd6149f))
- **webgpu:** a token's queries, keys and values in one native pass ([eecbc8b](https://github.com/webgllm/webgllm/commit/eecbc8b))
- draft batch costs carry over from reply to reply ([03c7b15](https://github.com/webgllm/webgllm/commit/03c7b15))
- **webgpu:** a native expanded Q6_K head, its bytes in RGBA32UI texels ([2a992e5](https://github.com/webgllm/webgllm/commit/2a992e5))
- **webgpu:** Q6_K head texels uploaded in the order the native head reads them ([87909c0](https://github.com/webgllm/webgllm/commit/87909c0))
- **webgpu:** the native Q6_K head runs in one K-slice ([e3154f2](https://github.com/webgllm/webgllm/commit/e3154f2))
- **webgpu:** the Q6_K word head as a native pass, its activations loaded once into workgroup memory ([5e1cda3](https://github.com/webgllm/webgllm/commit/5e1cda3))
- **webgpu:** a token's attention in chunks at any length where the GPU tuned it on (WebGPU's passes cost little) ([1e4aa88](https://github.com/webgllm/webgllm/commit/1e4aa88))
- **webgpu:** a token's add in one native pass (y's RMS, the residual, its sums of squares, the next matmul's input) ([539ae8e](https://github.com/webgllm/webgllm/commit/539ae8e))
- **webgpu:** a token's small passes that write disjoint regions of one target run as one dispatch; the Q6_K head reads its scales once a super-block ([4dbcddc](https://github.com/webgllm/webgllm/commit/4dbcddc))
- **load:** the GGUF loader reuses each layer's matrices ([f598c3f](https://github.com/webgllm/webgllm/commit/f598c3f))
- **load:** weights read in a worker of their own by default; read buffers recycled ([12ea326](https://github.com/webgllm/webgllm/commit/12ea326))
- **load:** free ONNX read buffers; optional reader worker and read-ahead ([459d849](https://github.com/webgllm/webgllm/commit/459d849))
- **load:** faster weight loading without dropped frames ([fe6f0ab](https://github.com/webgllm/webgllm/commit/fe6f0ab))

### ❤️ Thank You

- daveycodez

## 0.3.7 (2026-11-16)

### 🚀 Features

- tool calls held to their tools as they're written, in the model's own format, as llama.cpp's grammars hold them; a named tool is a required call to it ([3533bb4](https://github.com/webgllm/webgllm/commit/3533bb4))
- frequency_penalty, as llama.cpp's penalty_freq ([aa3cde0](https://github.com/webgllm/webgllm/commit/aa3cde0))
- continue a last assistant message, as llama.cpp does; past calls in the text only where a template can't write tool_calls ([ed69279](https://github.com/webgllm/webgllm/commit/ed69279))
- chat_template_kwargs, the reasoning effort to the template, and llama.cpp's thinking budget names ([f1455f9](https://github.com/webgllm/webgllm/commit/f1455f9))

### 🩹 Fixes

- a reference's siblings narrow it, definitions resolve against the tool, and each branch is checked on its own for strings ([1bdf539](https://github.com/webgllm/webgllm/commit/1bdf539))
- tool calls read and held by their schemas: XML strings stay text, Python calls need their required arguments, references resolve against the whole tool, a cut-off repair stays cut off ([1782818](https://github.com/webgllm/webgllm/commit/1782818))
- parallel_tool_calls defaults as llama.cpp's: on where the chat template writes a turn's second call ([98d35a2](https://github.com/webgllm/webgllm/commit/98d35a2))
- close schema objects as llama.cpp does: those that list properties ([88d1925](https://github.com/webgllm/webgllm/commit/88d1925))
- tool_choice none keeps the tools in the prompt, as llama.cpp does ([69da59b](https://github.com/webgllm/webgllm/commit/69da59b))
- pass messages and tools to the chat template as llama.cpp does ([d3d8589](https://github.com/webgllm/webgllm/commit/d3d8589))
- pass the whole tool history and tool results as given, as llama.cpp does; tool-call repair is opt-in ([6e9163b](https://github.com/webgllm/webgllm/commit/6e9163b))

### ❤️ Thank You

- daveycodez

## 0.3.6 (2026-11-15)

### 🩹 Fixes

- a required call is read from the answer as the reply reader reads it, thinking tags split across tokens too ([11acde2](https://github.com/webgllm/webgllm/commit/11acde2))
- cut-off calls by their own text, required calls only from a matched call in the answer, and no JSON answer beside a required call ([9c60d80](https://github.com/webgllm/webgllm/commit/9c60d80))
- top-p and min-p before the temperature, and the presence penalty on the whole window, as llama.cpp samples ([b68d7ce](https://github.com/webgllm/webgllm/commit/b68d7ce))
- **constraint:** whitespace runs as llama.cpp's grammar takes them; no stray-colon rule of webgllm's own ([c7ad450](https://github.com/webgllm/webgllm/commit/c7ad450))
- a call cut off by max_tokens comes back as written, with finish_reason length, as llama.cpp sends it ([71265a7](https://github.com/webgllm/webgllm/commit/71265a7))
- tool_choice required as llama.cpp has it: the call in the model's own format, and no end before it ([b9ec50b](https://github.com/webgllm/webgllm/commit/b9ec50b))
- **constraint:** whitespace after a value only before the end of what holds it, as llama.cpp's grammar takes it ([f968908](https://github.com/webgllm/webgllm/commit/f968908))
- **constraint:** an object with rules over its keys as a whole isn't held to an order ([b55cf08](https://github.com/webgllm/webgllm/commit/b55cf08))
- a batch of drafts checks each row with the penalties drawing them one by one would apply ([f94a19f](https://github.com/webgllm/webgllm/commit/f94a19f))
- **constraint:** whitespace optional after a key's colon, as in llama.cpp's grammar; a value right after it can't open with a stray colon ([f89e80d](https://github.com/webgllm/webgllm/commit/f89e80d))
- lengths pruned only where the walker reads the schema; a surrogate pair counts once; only own definitions inlined; identity in prefix file names ([88f1c14](https://github.com/webgllm/webgllm/commit/88f1c14))
- **constraint:** closed objects' keys in llama.cpp's order when generating (required as declared, then optional as declared); checking a parsed value takes any order ([26f8e34](https://github.com/webgllm/webgllm/commit/26f8e34))
- repetition and presence penalties as llama.cpp applies them: the last 64 tokens (repeat_last_n), prompt included, inside structured answers too ([1cdc75c](https://github.com/webgllm/webgllm/commit/1cdc75c))

### 🔥 Performance

- warm the constraint compiler when idle after a load; leave another identity's prefix files on disk instead of deleting them ([2c1aeb1](https://github.com/webgllm/webgllm/commit/2c1aeb1))
- **constraint:** $refs written in place, and string lengths held outside llguidance, where its masks were 27-34x slower ([0b06b9a](https://github.com/webgllm/webgllm/commit/0b06b9a))

### ❤️ Thank You

- daveycodez

## 0.3.5 (2026-11-15)

### 🚀 Features

- **webgllm:** keep several prompt prefixes, in memory and on disk ([#14](https://github.com/webgllm/webgllm/pull/14))

### ❤️ Thank You

- daveycodez

## 0.3.4 (2026-11-13)

### 🔥 Performance

- **webgllm:** first-load tuning that survives a busy GPU, and is shared by models of one shape ([#13](https://github.com/webgllm/webgllm/pull/13))

### ❤️ Thank You

- daveycodez

## 0.3.3 (2026-11-13)

### 🚀 Features

- **webgllm:** image input for LFM3-VL and LFM2.5-VL (GGUF and ONNX), and Qwen3.5's GGUF ([#12](https://github.com/webgllm/webgllm/pull/12))

### 🔥 Performance

- **webgllm:** Firefox on Android no longer stuck writing at 2 tokens/s, and reads prompts twice as fast ([41e907d](https://github.com/webgllm/webgllm/commit/41e907d))

### ❤️ Thank You

- daveycodez

## 0.3.2 (2026-11-12)

### 🔥 Performance

- **webgllm:** reading a prompt, waits on the GPU sleep instead of polling ([e46db93](https://github.com/webgllm/webgllm/commit/e46db93))

### ❤️ Thank You

- daveycodez

## 0.3.1 (2026-11-11)

### 🩹 Fixes

- **webgllm:** a worker of your own that fails no longer leaves requests waiting ([#11](https://github.com/webgllm/webgllm/pull/11))

### ❤️ Thank You

- daveycodez

## 0.3.0 (2026-11-11)

### 🚀 Features

- ⚠️  **webgllm:** a worker handler at the root, a worker of your own, tokens and logits ([#10](https://github.com/webgllm/webgllm/pull/10))

### ⚠️  Breaking Changes

- **webgllm:** a worker handler at the root, a worker of your own, tokens and logits  ([#10](https://github.com/webgllm/webgllm/pull/10))
  the worker option is a Worker, not () => Worker; the webgllm/advanced entry and the engine's exports are removed."
  M	apps/web/content/docs/guides/chat-and-streaming.mdx
  M	apps/web/content/docs/guides/loading-models.mdx
  D	apps/web/content/docs/reference/advanced.mdx
  M	apps/web/content/docs/reference/how-it-works.mdx
  M	apps/web/content/docs/reference/meta.json
  M	apps/web/content/docs/reference/webgllm.mdx
  M	apps/web/content/docs/requirements.mdx
  M	apps/web/src/components/demo/model.ts
  M	packages/webgllm/README.md
  M	packages/webgllm/bench/README.md
  M	packages/webgllm/bench/check-worker.ts
  A	packages/webgllm/bench/logits-worker.ts
  A	packages/webgllm/bench/logits.html
  A	packages/webgllm/bench/logits.ts
  M	packages/webgllm/bench/mem-worker.ts
  A	packages/webgllm/bench/own-worker.html
  A	packages/webgllm/bench/own-worker.ts
  M	packages/webgllm/bench/webgl-perf.ts
  M	packages/webgllm/package.json
  D	packages/webgllm/src/advanced.ts
  M	packages/webgllm/src/chat-completions.test.ts
  M	packages/webgllm/src/chat-completions.ts
  M	packages/webgllm/src/client.test.ts
  M	packages/webgllm/src/client.ts
  M	packages/webgllm/src/engine.ts
  M	packages/webgllm/src/fake-worker.test.ts
  A	packages/webgllm/src/handler.ts
  M	packages/webgllm/src/index.ts
  M	packages/webgllm/src/manifest.test.ts
  M	packages/webgllm/src/model-files.test.ts
  M	packages/webgllm/src/openai.ts
  M	packages/webgllm/src/pool.test.ts
  M	packages/webgllm/src/pool.ts
  M	packages/webgllm/src/prompt-images.ts
  M	packages/webgllm/src/prompt.ts
  M	packages/webgllm/src/protocol.ts
  M	packages/webgllm/src/sampler.test.ts
  M	packages/webgllm/src/sampler.ts
  M	packages/webgllm/src/shared-download.test.ts
  M	packages/webgllm/src/shared-download.ts
  M	packages/webgllm/src/storage.ts
  M	packages/webgllm/src/tokenizer-vocab.test.ts
  M	packages/webgllm/src/tokenizer.ts
  M	packages/webgllm/src/webgllm.test.ts
  M	packages/webgllm/src/webgllm.ts
  M	packages/webgllm/src/worker-cancellation.test.ts
  A	packages/webgllm/src/worker-download.test.ts
  M	packages/webgllm/src/worker-generation.test.ts
  M	packages/webgllm/src/worker-load.test.ts
  M	packages/webgllm/src/worker.ts

### ❤️ Thank You

- daveycodez

## 0.2.3 (2026-11-10)

### 🚀 Features

- **webgllm:** image input for Qwen3.5 ([#9](https://github.com/webgllm/webgllm/pull/9))

### ❤️ Thank You

- daveycodez

## 0.2.2 (2026-11-10)

### 🚀 Features

- **webgllm:** models by URL ([#8](https://github.com/webgllm/webgllm/pull/8))

### ❤️ Thank You

- daveycodez

## 0.2.1 (2026-11-09)

### 🚀 Features

- ⚠️  **webgllm:** OpenRouter's reasoning object for the effort and thinking budget ([#7](https://github.com/webgllm/webgllm/pull/7))

### ⚠️  Breaking Changes

- **webgllm:** OpenRouter's reasoning object for the effort and thinking budget  ([#7](https://github.com/webgllm/webgllm/pull/7))

### ❤️ Thank You

- daveycodez

## 0.2.0 (2026-11-09)

### 🚀 Features

- ⚠️  **webgllm:** an OpenAI-shaped client, and reasoning levels as reasoning_effort values ([#6](https://github.com/webgllm/webgllm/pull/6))

### ⚠️  Breaking Changes

- **webgllm:** an OpenAI-shaped client, and reasoning levels as reasoning_effort values  ([#6](https://github.com/webgllm/webgllm/pull/6))
  a model's reasoning levels hold the reasoning_effort value
  to send for each, { off: "none", high: "high" }, instead of the chat
  template's enable_thinking values ("false", "true"), which webgllm never
  read. They now describe webgllm's own API.
  * fix(webgllm): manifests stored before 0.2.0 read with today's reasoning levels; baseURL's comment says what it covers
  * chore: update every dependency to its latest version
  @huggingface/jinja 0.5.11, Nx 24.0.5, assistant-ui, fumadocs, TanStack
  Start, Tailwind and the rest. fumadocs-mdx stays at 15.5.1: 15.5.2 hands
  Vite its meta.json files as JavaScript, and the docs site's build fails.
  * fix(webgllm): a manifest from before 0.2.0 is deleted and its model resolves again, instead of being converted"
  M	apps/web/package.json
  M	package.json
  M	packages/webgllm/package.json
  M	packages/webgllm/src/lfm2-gguf.test.ts
  M	packages/webgllm/src/manifest.test.ts
  M	packages/webgllm/src/model-meta-types.test.ts
  M	packages/webgllm/src/model-meta.ts
  M	packages/webgllm/src/resolve-model-files.test.ts
  M	packages/webgllm/src/resolve-model.ts
  M	packages/webgllm/src/storage.ts
  M	packages/webgllm/src/webgllm.test.ts
  M	packages/webgllm/src/webgllm.ts
  M	pnpm-lock.yaml

### ❤️ Thank You

- daveycodez

## 0.1.9 (2026-11-09)

### 🩹 Fixes

- **webgllm:** count the prompt's tokens in the repetition penalty, as Transformers.js does ([#5](https://github.com/webgllm/webgllm/pull/5))

### ❤️ Thank You

- daveycodez

## 0.1.8 (2026-11-09)

### 🩹 Fixes

- **webgllm:** load ONNX exports by graph structure, and float16 tied embeddings ([#4](https://github.com/webgllm/webgllm/pull/4))

### ❤️ Thank You

- daveycodez

## 0.1.7 (2026-11-09)

### 🚀 Features

- **webgllm:** load any Hugging Face model by name, the catalog as known defaults ([#3](https://github.com/webgllm/webgllm/pull/3))

### ❤️ Thank You

- daveycodez

## 0.1.6 (2026-11-08)

### 🚀 Features

- **webgllm:** add models.downloaded to check storage without network requests ([296251f](https://github.com/webgllm/webgllm/commit/296251f))

### ❤️ Thank You

- daveycodez

## 0.1.5 (2026-11-06)

### 🩹 Fixes

- **webgllm:** close and merge allOf objects so they're fast to constrain ([5122079](https://github.com/webgllm/webgllm/commit/5122079))

### ❤️ Thank You

- daveycodez

## 0.1.4 (2026-11-06)

### 🩹 Fixes

- **webgllm:** stop a load still waiting when its model is unloaded ([01d2b9a](https://github.com/webgllm/webgllm/commit/01d2b9a))

### ❤️ Thank You

- daveycodez

## 0.1.3 (2026-11-06)

### 🚀 Features

- **webgllm:** add onMemoryClaim so a host can count webgllm's memory in its own budget ([a3ed68e](https://github.com/webgllm/webgllm/commit/a3ed68e))

### ❤️ Thank You

- daveycodez

## 0.1.2 (2026-11-06)

### 🩹 Fixes

- **webgllm:** close objects only where it narrows the schema, scope refs by name ([b191958](https://github.com/webgllm/webgllm/commit/b191958))
- **webgllm:** cancel a queued resize without the reply, settle downloads on stored files ([13d0d13](https://github.com/webgllm/webgllm/commit/13d0d13))
- **bench:** count a frame gap across the first token, wait for uploads ([2cd750d](https://github.com/webgllm/webgllm/commit/2cd750d))
- **webgllm:** cap only formatting whitespace, not spaces inside strings ([50b67f8](https://github.com/webgllm/webgllm/commit/50b67f8))
- **webgllm:** self-check the q|k|v|gate projection the GPU computed ([3d93afc](https://github.com/webgllm/webgllm/commit/3d93afc))
- **webgllm:** keep a finished file whole under a second download, cap no-version retries ([426212e](https://github.com/webgllm/webgllm/commit/426212e))
- **webgllm:** end a stream's waiting read on return, size GGUFs without HEAD ([ff4455f](https://github.com/webgllm/webgllm/commit/ff4455f))
- **webgllm:** penalize only the reply's tokens, and keep thinking off when asked ([8179148](https://github.com/webgllm/webgllm/commit/8179148))
- **webgllm:** close response_format objects as tool schemas are ([5cafb26](https://github.com/webgllm/webgllm/commit/5cafb26))
- **webgllm:** let constrained JSON take the whitespace models write ([560279d](https://github.com/webgllm/webgllm/commit/560279d))
- **webgllm:** close a spent thinking budget the way the template does ([0138f5a](https://github.com/webgllm/webgllm/commit/0138f5a))

### ❤️ Thank You

- daveycodez

## 0.1.1 (2026-11-06)

### 🚀 Features

- **webgllm:** model lifecycle events and exact catalog types ([f338c9b](https://github.com/webgllm/webgllm/commit/f338c9b))

### 🩹 Fixes

- **webgllm:** serialize loads, fit startup probes, free failed readbacks ([b5f8530](https://github.com/webgllm/webgllm/commit/b5f8530))
- **bench:** start decode timing at the first token, survive bad requests ([b3bced0](https://github.com/webgllm/webgllm/commit/b3bced0))
- **webgllm:** serialize stream reads, scope tool schemas, refuse a zero penalty ([c724bfa](https://github.com/webgllm/webgllm/commit/c724bfa))
- **webgllm:** share file downloads and drop canceled evictions ([695dc9d](https://github.com/webgllm/webgllm/commit/695dc9d))
- **webgllm:** only resume downloads that match the stored version ([ccff39a](https://github.com/webgllm/webgllm/commit/ccff39a))

### 🔥 Performance

- **webgllm:** sample large top_k and nucleus without full scans ([c74f7e8](https://github.com/webgllm/webgllm/commit/c74f7e8))

### ❤️ Thank You

- daveycodez

## 0.1.0 (2026-11-06)

### 🚀 Features

- **webgllm:** repair automatic tool calls that fail their tools ([8a16dc0](https://github.com/webgllm/webgllm/commit/8a16dc0))
- **webgllm:** add the WebGLLM class with an OpenAI-compatible API ([88341fd](https://github.com/webgllm/webgllm/commit/88341fd))
- **webgllm:** add stop sequences, seeds, tool choice and downloads to the worker ([7208b59](https://github.com/webgllm/webgllm/commit/7208b59))
- enforce structured output across browser adapters ([7117df0](https://github.com/webgllm/webgllm/commit/7117df0))
- **checkers:** bound AI thinking and play forced moves immediately ([3b0a7cc](https://github.com/webgllm/webgllm/commit/3b0a7cc))
- **checkers:** add local AI game with model selection ([8d40ddd](https://github.com/webgllm/webgllm/commit/8d40ddd))
- **webgl:** setPromptBatch lets the caller trade prompt speed for memory; no f16 scratch without 4-token matmuls ([ad06f7f](https://github.com/webgllm/webgllm/commit/ad06f7f))
- **lofi-agent:** LFM3's GGUF on the WebGL engine is the default model ([751b9d7](https://github.com/webgllm/webgllm/commit/751b9d7))
- **webgl:** Qwen3.5-2B on the WebGL engine, from unsloth's Q4_0 GGUF and onnx-community's q4f16 ONNX ([2db28e0](https://github.com/webgllm/webgllm/commit/2db28e0))
- runtimes share a memory budget: loading a model unloads other runtimes' least recently used models until it fits ([ac80a85](https://github.com/webgllm/webgllm/commit/ac80a85))
- models download their latest version (main), and resume only from the same version ([1aa697d](https://github.com/webgllm/webgllm/commit/1aa697d))
- every adapter defaults to the model's own context window, read from the model; contextLength on all three ([7003626](https://github.com/webgllm/webgllm/commit/7003626))
- **webgl:** grow the cache to the model's own context length ([763e2e9](https://github.com/webgllm/webgllm/commit/763e2e9))
- **webgl:** run the LFM2.5 ONNX models, warm shaders up at load ([b6a2e75](https://github.com/webgllm/webgllm/commit/b6a2e75))
- **webgl:** speculative decoding by prompt lookup ([cc5a697](https://github.com/webgllm/webgllm/commit/cc5a697))
- **webgl:** an LFM3 inference engine and adapter on WebGL 2 ([2d78423](https://github.com/webgllm/webgllm/commit/2d78423))

### 🩹 Fixes

- **webgllm:** fail fast on values the constraint library can't prune ([c0e16e9](https://github.com/webgllm/webgllm/commit/c0e16e9))
- **webgl:** address pacing and calibration review findings ([2a52337](https://github.com/webgllm/webgllm/commit/2a52337))
- **webgl:** estimate cadence correctly with rounded timestamps ([9b40e53](https://github.com/webgllm/webgllm/commit/9b40e53))
- **webgl:** honor caller output limits and budgets for all thinking models ([7a713f5](https://github.com/webgllm/webgllm/commit/7a713f5))
- **webgl:** use recommended sampling defaults and remove game retries ([23f2491](https://github.com/webgllm/webgllm/commit/23f2491))
- **webgl:** the worker claims the textures it measured from the memory budget, not 1.1x the file ([57ac750](https://github.com/webgllm/webgllm/commit/57ac750))
- **webgl:** cleaning up old pinned copies no longer deletes a freshly downloaded tokenizer_config.json ([f7ce03f](https://github.com/webgllm/webgllm/commit/f7ce03f))
- **webgl:** loader checker passes on today's loaders: chunked embeddings, native Q6_K ([a5ab936](https://github.com/webgllm/webgllm/commit/a5ab936))
- **webgl:** closing the adapter stream stops generation; bench server path checks; loader checker reads raw weights; persistence only on downloads ([33ca06f](https://github.com/webgllm/webgllm/commit/33ca06f))
- **webgl:** review fixes: cut-down batches guard their rows, robust draft timing, background tuning that gives way ([e927621](https://github.com/webgllm/webgllm/commit/e927621))
- **webgl:** resume downloads after a closed handle, hidden-page waits, work-sized draws ([1fe4349](https://github.com/webgllm/webgllm/commit/1fe4349))
- **webgl:** free a model's GPU memory on switch, one batch size, portability fixes ([131dd88](https://github.com/webgllm/webgllm/commit/131dd88))
- **webgl:** attention masking with whole-vector ops ([685c70a](https://github.com/webgllm/webgllm/commit/685c70a))
- **webgl:** ask the GPU which multi-target draws it can do ([b61ad2d](https://github.com/webgllm/webgllm/commit/b61ad2d))
- **webgl:** code review fixes; drafting tries again mid-reply ([c04c116](https://github.com/webgllm/webgllm/commit/c04c116))
- **webgl:** pace long prompts so the browser keeps drawing ([d1838eb](https://github.com/webgllm/webgllm/commit/d1838eb))
- **webgl:** code review fixes ([c45fb50](https://github.com/webgllm/webgllm/commit/c45fb50))
- **webgl:** thinking, stops, lost contexts and downloads ([7c3c2db](https://github.com/webgllm/webgllm/commit/7c3c2db))

### 🔥 Performance

- **webgl:** improve slow fence polling and benchmark rendering ([24f02c2](https://github.com/webgllm/webgllm/commit/24f02c2))
- **webgl:** adapt pacing to slow GPU fences ([b9d9101](https://github.com/webgllm/webgllm/commit/b9d9101))
- **webgl:** recover prefill and reduce attention texture memory ([781cdd1](https://github.com/webgllm/webgllm/commit/781cdd1))
- **webgl:** shorten cold calibration and retire consumed model reads ([20316e5](https://github.com/webgllm/webgllm/commit/20316e5))
- **webgl:** skip redundant startup layout work ([f4d51e1](https://github.com/webgllm/webgllm/commit/f4d51e1))
- **webgl:** scale pacing from full workloads and stabilize startup selection ([8fe0a5a](https://github.com/webgllm/webgllm/commit/8fe0a5a))
- **webgl:** pace inference and reduce model loading memory ([4d7fef3](https://github.com/webgllm/webgllm/commit/4d7fef3))
- **webgl:** checkpoint stable structured-output prefixes ([26e1f67](https://github.com/webgllm/webgllm/commit/26e1f67))
- **webgl:** writing's attention panels two across, timed per GPU at load: 15-33% less attention time on an iPhone from 2k-8k keys ([bada4b5](https://github.com/webgllm/webgllm/commit/bada4b5))
- **webgl:** the tuner times the f16 matmul's choices on whole prompt batches, as prompts run them ([5c346e6](https://github.com/webgllm/webgllm/commit/5c346e6))
- **webgl:** writing's attention in short chunks laid out in panels: 2-2.5x less attention time from a 7,900-token cache on an M3 ([330c7f9](https://github.com/webgllm/webgllm/commit/330c7f9))
- **webgl:** f16 h2 reads by default, RGBA4 through the f16 matmul where the tuner times it faster, in 32 MiB chunks ([36655ce](https://github.com/webgllm/webgllm/commit/36655ce))
- **webgl:** foldEach off by default: it wrote Qwen3.5 GGUF 1.6% slower on an iPhone; fuseLinear and qInCache confirmed there ([4ba37a8](https://github.com/webgllm/webgllm/commit/4ba37a8))
- **webgl:** the f16 prompt matmul reads h2 texels: 4-11% faster prompts on byte weights, a pass and 15 MiB fewer ([f5b2319](https://github.com/webgllm/webgllm/commit/f5b2319))
- **webgl:** writing's queries go in free KV cache slots, drawn with the keys: one render pass fewer per attention layer ([1c1f097](https://github.com/webgllm/webgllm/commit/1c1f097))
- **webgl:** writing's Gated DeltaNet block in 6 render passes, not 10: Qwen3.5 writes 13-18% faster on an M3 ([dd4ab5e](https://github.com/webgllm/webgllm/commit/dd4ab5e))
- **webgl:** Qwen3.5 GGUF folds its norms into writing's matmuls: 34 render passes fewer a token ([2aee8fa](https://github.com/webgllm/webgllm/commit/2aee8fa))
- **webgl:** RGBA4 weights read in place in prompt matmuls: up to 6% faster prompts and 61-77 MiB less ([f6ddb7e](https://github.com/webgllm/webgllm/commit/f6ddb7e))
- **webgl:** the attention cache grows 512 positions at a time: 90 MiB less just past 8k tokens ([b87940b](https://github.com/webgllm/webgllm/commit/b87940b))
- **webgl:** Qwen3.5's follow-ups read 159 tokens in 90 ms, not 687 in 274: an anchor snapshot where they stop sharing the chat ([2e5668c](https://github.com/webgllm/webgllm/commit/2e5668c))
- **webgl:** Qwen3.5 takes 49-63 MiB less: no conv buffers, logits for one row ([f47c1a4](https://github.com/webgllm/webgllm/commit/f47c1a4))
- **webgl:** a first load takes 29-36 MiB less: the self-check frees its scratch ([4ba93dd](https://github.com/webgllm/webgllm/commit/4ba93dd))
- **webgl:** Qwen3.5 reads prompts 3.4-4.7x faster: the Gated DeltaNet recurrence runs in chunks of 32 tokens ([6143693](https://github.com/webgllm/webgllm/commit/6143693))
- **webgl:** GGUF writes as fast as ONNX: a Q6_K output head in word texels, split along K ([823428d](https://github.com/webgllm/webgllm/commit/823428d))
- **webgl:** faster loads: warm-up through one layer of each kind, tuning when idle ([24a033f](https://github.com/webgllm/webgllm/commit/24a033f))
- **webgl:** tuner times a whole layer at 512 rows, bar scales with noise; draft counts per reply ([914a78b](https://github.com/webgllm/webgllm/commit/914a78b))
- **webgl:** load-time tuning of how prompt matmuls split K, remembered per GPU ([0243bea](https://github.com/webgllm/webgllm/commit/0243bea))
- **webgl:** drafts by measured cost per token; guessing can come back; prompts skip the last layer's unused rows ([2800af0](https://github.com/webgllm/webgllm/commit/2800af0))
- **webgl:** retries resume from the prompt's saved end; greedy reads back only token ids; GGUF heads pick all rows at once ([39ab8e5](https://github.com/webgllm/webgllm/commit/39ab8e5))
- **webgl:** ready once the weights are in; shaders compile after ([c782d61](https://github.com/webgllm/webgllm/commit/c782d61))
- **webgl:** faster loads from storage: remembered checks, raw ONNX quants into RGBA4 ([82aca74](https://github.com/webgllm/webgllm/commit/82aca74))
- **webgl:** RGBA4 layer weights' scales and zero points in one texel too ([7471cd7](https://github.com/webgllm/webgllm/commit/7471cd7))
- **webgl:** pick the weight layout per GPU by timing both at load ([e117d81](https://github.com/webgllm/webgllm/commit/e117d81))
- **webgl:** 4-bit layer weights in byte texels; scales and zero points in one texel ([834ea35](https://github.com/webgllm/webgllm/commit/834ea35))
- **webgl:** split matmuls' partial sums as halves ([ebb43a9](https://github.com/webgllm/webgllm/commit/ebb43a9))
- **webgl:** pick's argmax from the chunk maxima's positions, constant-bound pick loops ([43c34a9](https://github.com/webgllm/webgllm/commit/43c34a9))
- **webgl:** up to 7 drafts per batch; review fixes ([bf7b127](https://github.com/webgllm/webgllm/commit/bf7b127))
- **webgl:** wllama-style drafts (4-token matches, longer drafts for longer matches, backoff), blocking reads with nothing to overlap ([2fdcfea](https://github.com/webgllm/webgllm/commit/2fdcfea))
- **webgl:** scissor each pass to its rect (render-area hint for tile-based GPUs) ([fe0f9a2](https://github.com/webgllm/webgllm/commit/fe0f9a2))
- **webgl:** prompt matmuls on f16-dequantized weights, 1.3-1.6x faster prompts on PowerVR ([5f13744](https://github.com/webgllm/webgllm/commit/5f13744))
- **webgl:** flush every layer while writing; bandwidth probe, chained matmul bench ([3d69724](https://github.com/webgllm/webgllm/commit/3d69724))
- **webgl:** long-context attention in 3 passes, work-based pacing ([e978612](https://github.com/webgllm/webgllm/commit/e978612))
- **webgl:** few-row conv blocks draw their output and state in one pass ([d56f963](https://github.com/webgllm/webgllm/commit/d56f963))
- **webgl:** attention fragments take two heads of a KV head together ([bdb0a5c](https://github.com/webgllm/webgllm/commit/bdb0a5c))
- **webgl:** the top-50 pick adds the head's slices itself ([4015681](https://github.com/webgllm/webgllm/commit/4015681))
- **webgl:** chunk writing's attention only past 1,024 keys ([31a10f5](https://github.com/webgllm/webgllm/commit/31a10f5))
- **webgl:** fold layer 0's and the head's norms when writing ([dede47c](https://github.com/webgllm/webgllm/commit/dede47c))
- **webgl:** guess from the chunk maxima, 128-query attention slices ([e2a602d](https://github.com/webgllm/webgllm/commit/e2a602d))
- **webgl:** bigger prompt batches for smaller models ([339d62c](https://github.com/webgllm/webgllm/commit/339d62c))
- **webgl:** chunked attention for long key ranges ([9715318](https://github.com/webgllm/webgllm/commit/9715318))
- **webgl:** 64-query attention slices on desktop, 4-slice cap ([c4485f7](https://github.com/webgllm/webgllm/commit/c4485f7))
- **webgl:** four tokens per fragment in prompt matmuls ([cbfb53c](https://github.com/webgllm/webgllm/commit/cbfb53c))
- **webgl:** check a draft batch in one head pass and one readback ([43d50e7](https://github.com/webgllm/webgllm/commit/43d50e7))
- **webgl:** constant split counts, one K/V cache texture, faster attend ([f1adec7](https://github.com/webgllm/webgllm/commit/f1adec7))
- **webgl:** drop the norm pass when writing, memoize kernel sources ([e38846e](https://github.com/webgllm/webgllm/commit/e38846e))
- **webgl:** pipeline writing, faster output head and small passes ([a31e0ff](https://github.com/webgllm/webgllm/commit/a31e0ff))
- **webgl:** spread row sums over 64 fragments, A/B timing on the bench ([634231b](https://github.com/webgllm/webgllm/commit/634231b))
- **webgl:** K-split narrow prompt matmuls, fewer render passes ([5e61dfe](https://github.com/webgllm/webgllm/commit/5e61dfe))
- **webgl:** RGBA4 weights and two-token prompt matmuls ([cb4c6b3](https://github.com/webgllm/webgllm/commit/cb4c6b3))
- **webgl:** share the Q6_K head's decoded activations across a fragment's rows ([d5ba3d1](https://github.com/webgllm/webgllm/commit/d5ba3d1))
- **webgl:** less memory, and the GGUF's Q6_K head read natively ([c09e941](https://github.com/webgllm/webgllm/commit/c09e941))

### ❤️ Thank You

- daveycodez