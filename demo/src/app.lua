local sched, err = require("cgemma").scheduler(config().scheduler)
if not sched then
  ngx.log(ngx.ERR, "cgemma error: ", err)
end

function gemma_inst()
  if not worker_gemma_inst then
    local gemma_cfg = config().gemma
    gemma_cfg.scheduler = sched
    local gemma, err = require("cgemma").new(gemma_cfg)
    if not gemma then
      ngx.log(ngx.ERR, "cgemma error: ", err)
      ngx.exit(ngx.HTTP_INTERNAL_SERVER_ERROR)
    end
    worker_gemma_inst = gemma
  end
  return worker_gemma_inst
end

function gemma_loop()
  local session, err = gemma_inst():session(config().session)
  if not session then
    ngx.log(ngx.ERR, "cgemma error: ", err)
    return ngx.HTTP_INTERNAL_SERVER_ERROR
  end
  local ok, err = session:load("dump.bin")
  if not ok then
    ngx.log(ngx.ERR, "cgemma error: ", err)
  end
  local ws, err = require("resty.websocket.server"):new(config().websocket)
  if not ws then
    ngx.log(ngx.ERR, "websocket error: ", err)
    return ngx.HTTP_CLOSE
  end
  local bytes, err = ws:send_text(require("cjson.safe").encode({
    role = "system",
    text = "New chat session started!"
  }))
  if not bytes then
    ngx.log(ngx.ERR, "websocket error: ", err)
    return ngx.OK
  end
  while session:ready() do
    local data, tp, err = ws:recv_frame()
    if tp == "text" then
      local msg = require("cjson.safe").decode(data)
      if not msg or not msg.role then
        ngx.log(ngx.ERR, "protocol error: unknown format")
        ws:send_close()
        return ngx.OK
      end
      if msg.role == "user" then
        if msg.text then
          local ok, err = session(msg.text, function(token, pos, prompt_size)
            local bytes, err = ws:send_text(require("cjson.safe").encode({
              role = "gemma",
              token = token,
              pos = pos,
              prompt_size = prompt_size
            }))
            if not bytes then
              ngx.log(ngx.ERR, "websocket error: ", err)
              return false
            end
            return true
          end)
          if not ok then
            ngx.log(ngx.ERR, "cgemma error: ", err)
            ws:send_close()
            return ngx.OK
          end
        end
      else
        local bytes, err = ws:send_text(require("cjson.safe").encode({
          role = "system",
          text = "Unsupported role!"
        }))
        if not bytes then
          ngx.log(ngx.ERR, "websocket error: ", err)
          return ngx.OK
        end
      end
    elseif tp == "ping" then
      local bytes, err = wb:send_pong()
      if not bytes then
        ngx.log(ngx.ERR, "websocket error: ", err)
        return ngx.OK
      end
    elseif tp == "close" then
      return ngx.OK
    elseif tp ~= "pong" then
      if err then
        ngx.log(ngx.ERR, "websocket error: ", err)
      end
      ws:send_close()
      return ngx.OK
    end
  end
  ws:send_text(require("cjson.safe").encode({
    role = "system",
    text = "Exceed the maximum number of tokens!"
  }))
  return ngx.OK
end
