-- Parse cli-args
local args = {}
for i, v in ipairs(arg) do
  if string.sub(v, 1, 2) == "--" then
    if arg[i + 1] and string.sub(arg[i + 1], 1, 2) ~= "--" then
      args[string.sub(v, 3)] = arg[i + 1]
    else
      args[string.sub(v, 3)] = true
    end
  end
end
if args.help then
  require("cgemma").info()
  print()
  print("Store prompt's KV cache in a file.")
  print()
  print("Usage: cat /path/to/prompt.txt | resty dump_prompt.lua [options]")
  print()
  print("Available options:")
  print("  --tokenizer: Path of tokenizer model file. (default: tokenizer.spm)")
  print("  --weights: Path of model weights file. (default: 4b-it-sfp.sbs)")
  print("  --map: Enable memory-mapping? -1 = auto, 0 = no, 1 = yes. (default: -1)")
  print("  --seq_len: Sequence length, capped by max context window of model. (default: 8192)")
  print("  --max_generated_tokens: Maximum number of tokens to generate. (default: 2048)")
  print("  --prefill_tbatch: Maximum batch size during prefill phase (default: 256)")
  print("  --temperature: Temperature for top-K. (default: 1.0)")
  print("  --top_k: Number of top-K tokens to sample from. (default: 1)")
  print("  --image: Path of image file (PPM format: P6, binary).")
  print("  --kv_cache: Path of KV cache file.")
  print("  --gen: Enable generation phase.")
  print("  --output: Path of output file. (default: dump.bin)")
  print("  --num_threads: Maximum number of threads to use, 0 = unlimited. (default: 0)")
  print("  --pin: Pin threads? -1 = auto, 0 = no, 1 = yes. (default: -1)")
  print("  --bind: Bind memory to sockets? -1 = auto, 0 = no, 1 = yes. (defaut: -1)")
  print("  --skip_packages: Index of the first socket to use, 0 = unlimited. (default: 0)")
  print("  --max_packages: Maximum number of sockets to use, 0 = unlimited. (default: 0)")
  print("  --skip_clusters: Index of the first CCX to use, 0 = unlimited. (default: 0)")
  print("  --max_clusters: Maximum number of CCXs to use, 0 = unlimited. (default: 0)")
  print("  --skip_lps: Index of the first LP to use, 0 = unlimited. (default: 0)")
  print("  --max_lps: Maximum number of LPs to use, 0 = unlimited. (default: 0)")
  print("  --stats: Print statistics at end.")
  return
end

-- Config global scheduler
local ok, err = require("cgemma").scheduler.config({
  num_threads = tonumber(args.num_threads),
  pin = tonumber(args.pin),
  skip_packages = tonumber(args.skip_packages),
  max_packages = tonumber(args.max_packages),
  skip_clusters = tonumber(args.skip_clusters),
  max_clusters = tonumber(args.max_clusters),
  skip_lps = tonumber(args.skip_lps),
  max_lps = tonumber(args.max_lps)
})
if not ok then
  error("Opoos! "..err)
end

print("Loading model ...")
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  weights = args.weights or "4b-it-sfp.sbs",
  map = args.map
})
if not gemma then
  error("Opoos! "..err)
end

local image, err
if args.image then
  print("Embedding image ...")
  image, err = gemma:embed_image(args.image)
  if not image then
    error("Opoos! "..err)
  end
end

-- Create a session
local session, err = gemma:session({
  seq_len = tonumber(args.seq_len),
  max_generated_tokens = tonumber(args.max_generated_tokens),
  prefill_tbatch = tonumber(args.prefill_tbatch),
  temperature = tonumber(args.temperature),
  top_k = tonumber(args.top_k),
  no_wrapping = true
})
if not session then
  error("Opoos! "..err)
end
if args.kv_cache then
  -- Restore the previous session
  local ok, err = session:load(args.kv_cache)
  if ok then
    print("Previous session restored")
  else
    print("Opoos! "..err)
  end
end

print("Reading prompt ...")
local stream_fn
if args.gen then
  stream_fn = function(token, pos, prompt_size)
    if pos < prompt_size then
      io.write(string.format("%d / %d\r", pos + 1, prompt_size))
    elseif token then
      if pos == prompt_size then
        io.write("\nreply: ")
      end
      io.write(token)
    else
      print()
    end
    io.flush()
    return true
  end
else
  stream_fn = function(token, pos, prompt_size)
    if pos >= prompt_size then
      return false
    end
    io.write(string.format("%d / %d\r", pos + 1, prompt_size))
    io.flush()
    return true
  end
end
local ok, err
if image then
  ok, err = session(image, io.read("*a"), stream_fn)
else
  ok, err = session(io.read("*a"), stream_fn)
end
if not ok then
  error("Opoos! "..err)
end
print()

-- Dump the current session to "dump.bin"
local ok, err = session:dump(args.output or "dump.bin")
if not ok then
  error("Opoos! "..err)
end
print(string.format("Done! Session states of the prompt have been dumped to \"%s\"", args.output or "dump.bin"))
if args.stats then
  print("\n\nStatistics:\n")
  for k, v in pairs(session:stats()) do
    print("  "..k.." = "..v)
  end
end
