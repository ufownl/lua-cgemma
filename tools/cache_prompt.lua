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
  print("Usage: cat /path/to/prompt.txt | resty cache_prompt.lua [options]")
  print()
  print("Available options:")
  print("  --tokenizer: Path of tokenizer model file. (default: tokenizer.spm)")
  print("  --model: Model type (default: gemma2-2b-pt)")
  print("    2b-it = Gemma 2B parameters, instruction-tuned")
  print("    2b-pt = Gemma 2B parameters, pretrained")
  print("    7b-it = Gemma 7B parameters, instruction-tuned")
  print("    7b-pt = Gemma 7B parameters, pretrained")
  print("    gr2b-it = Griffin 2B parameters, instruction-tuned")
  print("    gr2b-pt = Griffin 2B parameters, pretrained")
  print("    gemma2-2b-it = Gemma2 2B parameters, instruction-tuned")
  print("    gemma2-2b-pt = Gemma2 2B parameters, pretrained")
  print("    9b-it = Gemma2 9B parameters, instruction-tuned")
  print("    9b-pt = Gemma2 9B parameters, pretrained")
  print("    27b-it = Gemma2 27B parameters, instruction-tuned")
  print("    27b-pt = Gemma2 27B parameters, pretrained")
  print("  --weights: Path of model weights file. (default: 2.0-2b-it-sfp.sbs)")
  print("  --weight_type: Weight type (default: sfp)")
  print("  --prefill_tbatch: Maximum batch size during prefill phase (default: 64)")
  print("  --output: Path of output file. (default: dump.bin)")
  print("  --num_threads: Maximum number of threads to use, 0 = unlimited. (default: 0)")
  print("  --pin: Pin threads? -1 = auto, 0 = no, 1 = yes. (default: -1)")
  print("  --skip_packages: Index of the first socket to use, 0 = unlimited. (default: 0)")
  print("  --max_packages: Maximum number of sockets to use, 0 = unlimited. (default: 0)")
  print("  --skip_clusters: Index of the first CCX to use, 0 = unlimited. (default: 0)")
  print("  --max_clusters: Maximum number of CCXs to use, 0 = unlimited. (default: 0)")
  print("  --skip_lps: Index of the first LP to use, 0 = unlimited. (default: 0)")
  print("  --max_lps: Maximum number of LPs to use, 0 = unlimited. (default: 0)")
  print("  --stats: Print statistics at end.")
  return
end

-- Create a scheduler instance
local sched, err = require("cgemma").scheduler({
  num_threads = args.num_threads,
  pin = args.pin,
  skip_packages = args.skip_packages,
  max_packages = args.max_packages,
  skip_clusters = args.skip_clusters,
  max_clusters = args.max_clusters,
  skip_lps = args.skip_lps,
  max_lps = args.max_lps
})
if not sched then
  print("Opoos! ", err)
  return
end

print("Loading model ...")
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  model = args.model or "gemma2-2b-pt",
  weights = args.weights or "2.0-2b-it-sfp.sbs",
  weight_type = args.weight_type,
  scheduler = sched
})
if not gemma then
  print("Opoos! ", err)
  return
end

-- Create a session
local session, seed = gemma:session({
  max_generated_tokens = 1,
  prefill_tbatch = args.prefill_tbatch
})
if not session then
  print("Opoos! ", seed)
  return
end
print("Random seed of session: ", seed)

print("Reading prompt ...")
local prompt = io.read("*a")
local ok, err = session(prompt, function(token, pos, prompt_size)
  if pos >= prompt_size then
    return false
  end
  io.write(string.format("%d / %d\r", pos + 1, prompt_size))
  io.flush()
  return true
end)
if not ok then
  print("Opoos! ", err)
  return
end
print()

-- Dump the current session to "dump.bin"
local ok, err = session:dump(args.output or "dump.bin")
if not ok then
  print("Opoos! ", err)
  return
end
print(string.format("Done! Session states of the prompt have been dumped to \"%s\"", args.output or "dump.bin"))
if args.stats then
  print("\n\nStatistics:\n")
  for k, v in pairs(session:stats()) do
    print("  "..k.." = "..v)
  end
end
