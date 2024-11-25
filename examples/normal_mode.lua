local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Normal mode chatbot demo.",
    "resty normal_mode.lua [options]"
  )
  print("  --image: Path of image file (PPM format: P6, binary).")
  print("  --kv_cache: Path of KV cache file.")
  print("  --stats: Print statistics at end of turn.")
  return
end

-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  model = args.model or "gemma2-2b-it",
  weights = args.weights or "2.0-2b-it-sfp.sbs",
  weight_type = args.weight_type
})
if not gemma then
  error("Opoos! "..err)
end

local image, err
if args.image then
  -- Load image data
  image, err = require("cgemma").image(args.image)
  if not image then
    error("Opoos! "..err)
  end
end

-- Create a chat session
local session, err = image and gemma:session(image, {top_k = 50}) or gemma:session({top_k = 50})
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
    local reply, err = session(text)
    if not reply then
      error("Opoos! "..err)
    end
    print("reply: ", reply)

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
