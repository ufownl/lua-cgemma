local args = require("argparse").parse(arg)
if args.help then
  require("argparse").help(
    "AI function demo.",
    "resty ai_function.lua [options]"
  )
  return
end

-- Create a Gemma instance
local gemma = assert(require("cgemma").new({
  tokenizer = args.tokenizer or "tokenizer.spm",
  weights = args.weights or "4b-it-sfp.sbs"
}))

local function implement(...)
  local args = {...}
  local func_names = {}
  local queries = {}
  for i, v in ipairs(args) do
    if i % 2 == 0 then
      local func_name = assert(
        string.match(args[i - 1], "^def%s+([a-zA-Z0-9_]+)"),
        "Opoos! Bad function declaration."
      )
      table.insert(func_names, func_name)
      table.insert(func_names, "")
      table.insert(func_names, "")
      table.insert(queries, assert(gemma:session({max_generated_tokens = 1})))
      table.insert(queries, string.format("You are now the following python function:\n```python\n%s\n    '''%s'''\n```\n\nWhen you are called later, please reply with your return value only.", args[i - 1], v))
      table.insert(queries, function(token, pos, prompt_size)
        return pos < prompt_size
      end)
    end
  end
  local result = assert(require("cgemma").batch(unpack(queries)))
  local funcs = {}
  for i = 1, #queries, 3 do
    local func_name = func_names[i]
    local context = assert(queries[i]:dumps())
    table.insert(funcs, function(...)
      local session = assert(gemma:session({top_k = 5}))
      session:loads(context)
      local args = {...}
      for i, v in ipairs(args) do
        if type(v) == "nil" then
          args[i] = "None"
        elseif type(v) == "string" then
          args[i] = "'''"..v.."'''"
        end
      end
      return assert(session(string.format("%s(%s)", func_name, table.concat(args, ", "))))
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
