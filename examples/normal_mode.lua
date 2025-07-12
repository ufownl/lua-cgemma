local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Normal mode chatbot demo.",
    "resty normal_mode.lua [options]"
  )
  print("  --image: Path of image file (PPM format: P6, binary).")
  print("  --seq_len: Sequence length, capped by max context window of model. (default: 8192)")
  print("  --max_generated_tokens: Maximum number of tokens to generate. (default: 2048)")
  print("  --temperature: Temperature for top-K. (default: 1.0)")
  print("  --top_k: Number of top-K tokens to sample from. (default: 5)")
  print("  --kv_cache: Path of KV cache file.")
  print("  --stats: Print statistics at end of turn.")
  return
end

-- Create a Gemma instance
local gemma = assert(require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  weights = args.weights or "4b-it-sfp.sbs"
}))

local image
if args.image then
  -- Load image data
  image = assert(gemma:embed_image(args.image))
end

-- Create a chat session
local session = assert(gemma:session({
  seq_len = tonumber(args.seq_len),
  max_generated_tokens = tonumber(args.max_generated_tokens),
  temperature = tonumber(args.temperature),
  top_k = tonumber(args.top_k) or 5
}))

while true do
  if args.kv_cache then
    -- Restore the previous session
    local ok, err = session:load(args.kv_cache)
    if ok then
      print("Previous conversation restored")
    else
      print("New conversation started")
    end
  else
    print("New conversation started")
  end

  -- Multi-turn chat
  while session:ready() do
    io.write("> ")
    local text = io.read()
    if not text then
      if args.kv_cache then
        print("End of file, dumping current session ...")
        -- Dump the current session
        assert(session:dump(args.kv_cache))
      end
      print("Done")
      return
    end
    print("reply: ", assert(session(unpack(image and {image, text} or {text}))))

    if args.stats then
      print("\n\nStatistics:\n")
      for k, v in pairs(session:stats()) do
        print("  "..k.." = "..v)
      end
    end
  end

  print("Exceed the maximum number of tokens")
  session:reset()
end
