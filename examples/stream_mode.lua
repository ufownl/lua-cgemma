local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Stream mode chatbot demo.",
    "resty stream_mode.lua [options]"
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
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  weights = args.weights or "4b-it-sfp.sbs"
})
if not gemma then
  error("Opoos! "..err)
end

local image, err
if args.image then
  -- Load image data
  image, err = gemma:embed_image(args.image)
  if not image then
    error("Opoos! "..err)
  end
end

-- Create a chat session
local session, err = gemma:session({
  seq_len = tonumber(args.seq_len),
  max_generated_tokens = tonumber(args.max_generated_tokens),
  temperature = tonumber(args.temperature),
  top_k = tonumber(args.top_k) or 5
})
if not session then
  error("Opoos! "..err)
end

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
        local ok, err = session:dump(args.kv_cache)
        if not ok then
          error("Opoos! "..err)
        end
      end
      print("Done")
      return
    end
    local t = image and {image} or {}
    table.insert(t, text)
    table.insert(t, function(token, pos, prompt_size)
      if pos < prompt_size then
        -- Gemma is processing the prompt
        io.write(pos == 0 and "reading and thinking ." or ".")
      elseif token then
        -- Stream the token text output by Gemma here
        if pos == prompt_size then
          io.write("\nreply: ")
        end
        io.write(token)
      else
        -- Gemma's output reaches the end
        print()
      end
      io.flush()
      -- return `true` indicates success; return `false` indicates failure and terminates the generation
      return true
    end)
    local ok, err = session(unpack(t))
    if not ok then
      error("Opoos! "..err)
    end

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
