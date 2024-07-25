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
  print("  --num_threads: Number of threads in scheduler. (default: hardware concurrency)")
  print("  --tokenizer: Path of tokenizer model file. (default: tokenizer.spm)")
  print("  --model: Model type (default: 2b-pt)")
  print("    2b-it = 2B parameters, instruction-tuned")
  print("    2b-pt = 2B parameters, pretrained")
  print("    7b-it = 7B parameters instruction-tuned")
  print("    7b-pt = 7B parameters, pretrained")
  print("    9b-it = 9B parameters instruction-tuned")
  print("    9b-pt = 9B parameters, pretrained")
  print("    27b-it = 27B parameters instruction-tuned")
  print("    27b-pt = 27B parameters, pretrained")
  print("    gr2b-it = griffin 2B parameters, instruction-tuned")
  print("    gr2b-pt = griffin 2B parameters, pretrained")
  print("  --weights: Path of model weights file. (default: 2b-it-sfp.sbs)")
  print("  --weight_type: Weight type (default: sfp)")
  print("  --max_tokens: Maximum number of tokens (default: 3072)")
  print("  --prefill_tbatch: Maximum batch size during prefill phase (default: 64)")
  print("  --output: Path of output file. (default: dump.bin)")
  print("  --stats: Print statistics at end.")
  return
end

-- Create a scheduler instance
local sched, err = require("cgemma").scheduler(tonumber(args.num_threads))
if not sched then
  print("Opoos! ", err)
  return
end

print("Loading model ...")
-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  model = args.model or "2b-pt",
  weights = args.weights or "2b-it-sfp.sbs",
  weight_type = args.weight_type,
  scheduler = sched
})
if not gemma then
  print("Opoos! ", err)
  return
end

-- Create a session
local session, seed = gemma:session({
  max_tokens = args.max_tokens,
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
