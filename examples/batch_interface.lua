local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "Batch normal demo.",
    "resty batch_normal.lua [options]"
  )
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
  print("Opoos! ", err)
  return
end

-- Create 3 chat sessions
local sessions = {}
for i = 1, 3 do
  local session, err = gemma:session()
  if not session then
    print("Opoos! ", err)
    return
  end
  table.insert(sessions, session)
end

-- Define callback function for stream mode
local function stream(prefix, token, pos, prompt_size)
  if pos < prompt_size then
    -- Gemma is processing the prompt
    io.write(pos == 0 and prefix.."reading and thinking ." or ".")
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
end

-- Run multiple queries using batch interface
local queries = {
  {
    sessions[1], "Tell me 1+1=?",
    sessions[2], "Hello, world!",
    sessions[3], "Hey, man!", function(token, pos, prompt_size)
      return stream("Turn 1, Q3: Hey, man!\n", token, pos, prompt_size)
    end
  },
  {
    sessions[1], "Write it using Python.", function(token, pos, prompt_size)
      return stream("Turn 2, Q1: Write it using Python.\n", token, pos, prompt_size)
    end,
    sessions[2], "Print what I said using Python.",
    sessions[3], "Write what I said in lowercase."
  }
}
for i, query in ipairs(queries) do
  print(string.format("Turn %d:\n", i))

  -- Make a batch call
  local result, err = require("cgemma").batch(unpack(query))
  if not result then
    print("Opoos! ", err)
    return
  end

  -- Display the result of this batch call
  local idx = 1
  for j = 1, #query do
    if type(query[j]) == "string" then
      print(string.format("Q%d: %s\n", idx, query[j]))
      local resp, err = result(query[j - 1])
      if resp then
        print(resp)
      else
        print("Opoos! ", err)
      end
      idx = idx + 1
    end
  end

  if args.stats then
    print("\n\nStatistics:\n")
    for k, v in pairs(result:stats()) do
      print("  "..k.." = "..v)
    end
  end
  print()
end
