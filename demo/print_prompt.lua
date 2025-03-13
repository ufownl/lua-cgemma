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
  print("Print prompt for online demo.")
  print()
  print("Usage: resty print_prompt.lua [options]")
  print()
  print("Available options:")
  print("  --brief: Print the brief version.")
  return
end

local template, err = io.open("prompt_template.md")
if not template then
  error(err)
end
local readme, err = io.open("../README.md")
if not readme then
  template:close()
  error(err)
end
readme:read("*l")  -- Skip the line of badges
local content = ""
if args.brief then
  local state = 0
  while true do
    local line = readme:read("*l")
    if not line then
      break
    end
    if state == 0 then
      content = content..line.."\n"
      if line == "## Usage" then
        state = 1
      end
    elseif state == 1 then
      if line == "## License" then
        content = content.."\nPlease see [README.md](https://github.com/ufownl/lua-cgemma/blob/stable/README.md) for more details.\n\n"..line.."\n"
        state = 2
      end
    else
      content = content..line.."\n"
    end
  end
else
  content = readme:read("*a")
end
io.write(string.format(string.sub(template:read("*a"), 1, -2), content))
readme:close()
template:close()
