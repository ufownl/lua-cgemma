local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "AI function demo.",
    "resty ai_function.lua [options]"
  )
  return
end

-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  model = args.model or "gemma3-4b",
  weights = args.weights or "4b-it-sfp.sbs",
  weight_type = args.weight_type
})
if not gemma then
  error("Opoos! "..err)
end

local function implement(...)
  local args = {...}
  local func_names = {}
  local queries = {}
  for i, v in ipairs(args) do
    if i % 2 == 0 then
      local func_name = string.match(args[i - 1], "^def%s+([a-zA-Z0-9_]+)")
      if not func_name then
        error("Opoos! Bad function declaration.")
      end
      table.insert(func_names, func_name)
      table.insert(func_names, "")
      table.insert(func_names, "")
      local session, err = gemma:session({max_generated_tokens = 1})
      if not session then
        error("Opoos! "..err)
      end
      table.insert(queries, session)
      table.insert(queries, string.format("You are now the following python function:\n```python\n%s\n    '''%s'''\n```\n\nWhen you are called later, please reply with your return value only.", args[i - 1], v))
      table.insert(queries, function(token, pos, prompt_size)
        return pos < prompt_size
      end)
    end
  end
  local result, err = require("cgemma").batch(unpack(queries))
  if not result then
    error("Opoos! "..err)
  end
  local funcs = {}
  for i = 1, #queries, 3 do
    local func_name = func_names[i]
    local context, err = queries[i]:dumps()
    if not context then
      error("Opoos! "..err)
    end
    table.insert(funcs, function(...)
      local session, err = gemma:session({top_k = 5})
      if not session then
        error("Opoos! "..err)
      end
      session:loads(context)
      local args = {...}
      for i, v in ipairs(args) do
        if type(v) == "nil" then
          args[i] = "None"
        elseif type(v) == "string" then
          args[i] = "'''"..v.."'''"
        end
      end
      local reply, err = session(string.format("%s(%s)", func_name, table.concat(args, ", ")))
      if not reply then
        error("Opoos! "..err)
      end
      return reply
    end)
  end
  return unpack(funcs)
end

print("Implementing `reverse`, `multiply`, `fake_people` ...")
local reverse, multiply, fake_people = implement(
  "def reverse(s: str) -> str:",
  "Reverse the given string.",

  "def multiply(a: int, b: int) -> int:",
  "Multiply the given two integers.",

  "def fake_people(n: int) -> list[dict]:",
  "Generates n different examples of fake data representing people, each with a name, a gender, and an age."
)

print("Calling `reverse(\"Hello, world!\")` ...")
print(reverse("Hello, world!"))

print("Calling `multiply(6, 7)` ...")
print(multiply(6, 7))

print("Calling `fake_people(4)` ...")
print(fake_people(4))
