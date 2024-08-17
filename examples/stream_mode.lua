local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Stream mode chatbot demo.",
    "resty stream_mode.lua [options]"
  )
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

-- Create a chat session
local session, err = gemma:session()
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
    local ok, err = session(text, function(token, pos, prompt_size)
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
