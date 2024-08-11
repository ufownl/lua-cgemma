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

local queries = {
  {
    sessions[1], "Tell me 1+1=?",
    sessions[2], "Hello, world!",
    sessions[3], "Hey, man!"
  },
  {
    sessions[1], "Write it using Python.",
    sessions[2], "Print what I said using Python.",
    sessions[3], "Write what I said in lowercase."
  }
}
for i, query in ipairs(queries) do
  -- Run multiple queries in normal mode using batch interface
  print(string.format("Turn %d:\n", i))
  local result, err = require("cgemma").batch(unpack(query))
  if not result then
    print("Opoos! ", err)
    return
  end
  for j = 1, #query, 2 do
    print(string.format("Q: %s\n", query[j + 1]))
    local resp, err = result(query[j])
    if resp then
      print(resp)
    else
      print("Opoos! ", err)
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
