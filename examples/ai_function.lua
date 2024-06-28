local args = require("argparse").parse(arg)
if args["help"] then
  require("argparse").help(
    "AI function demo.",
    "resty ai_function.lua [options]"
  )
  return
end

-- Create a Gemma instance
local gemma, err = require("cgemma").new({
  tokenizer = args["tokenizer"] or "tokenizer.spm",
  model = args["model"] or "2b-it",
  weights = args["weights"] or "2b-it-sfp.sbs",
  weight_type = args["weight_type"]
})
if not gemma then
  print("Opoos! ", err)
  return
end

local function implement(declaration, description)
  local name = string.match(declaration, "^def%s+([a-zA-Z0-9_]+)")
  if not name then
    return nil, "Bad function declaration."
  end
  local session, err = gemma:session({
    max_generated_tokens = 1
  })
  if not session then
    print("Opoos! ", err)
    return
  end
  local ok, err = session(string.format("You are now the following python function: ```# %s\n%s```\n\nOnly respond with your `return` value. Do not include any other explanatory text in your response.", description, declaration), function(token, pos, prompt_size)
    return pos < prompt_size
  end)
  if not ok then
    return nil, err
  end
  local context, err = session:dumps()
  if not context then
    return nil, err
  end
  return function(...)
    local session, err = gemma:session()
    if not session then
      print("Opoos! ", err)
      return
    end
    session:loads(context)
    local args = {...}
    local text = ""
    for i, v in ipairs(args) do
      text = text..", "..(type(v) ~= nil and v or "None")
    end
    return session(string.format("%s(%s)", name, string.sub(text, 3)))
  end
end

print("Implementing `fake_people` ...")
local fake_people, err = implement(
  "def fake_people(n: int) -> list[dict]:",
  "Generates n different examples of fake data representing people, each with a name, a gender, and an age."
)
if not fake_people then
  print("Opoos! ", err)
  return
end
print("Implementing `multiply` ...")
local multiply, err = implement(
  "def multiply(a: int, b: int) -> int:",
  "Multiply the given two integers."
)
if not multiply then
  print("Opoos! ", err)
  return
end
print("Calling `fake_people(4)` ...")
local resp, err = fake_people(4)
if not resp then
  print("Opoos! ", err)
  return
end
print(resp)
print("Calling `multiply(2, 9)` ...")
local resp, err = multiply(2, 9)
if not resp then
  print("Opoos! ", err)
  return
end
print(resp)
print("Calling `fake_people(8)` ...")
local resp, err = fake_people(8)
if not resp then
  print("Opoos! ", err)
  return
end
print(resp)
