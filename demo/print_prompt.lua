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
io.write(string.format(string.sub(template:read("*a"), 1, -2), readme:read("*a")))
readme:close()
template:close()
