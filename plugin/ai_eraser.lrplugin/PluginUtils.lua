--[[----------------------------------------------------------------------------
PluginUtils.lua
Utility functions for Efface Magique LR
Path discovery, environment resolution, and cross-platform command execution
------------------------------------------------------------------------------]]

local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'
local LrShell = import 'LrShell'
local LrTasks = import 'LrTasks'

local PluginUtils = {}

--- Get the plugin directory path
function PluginUtils.getPluginDir()
    return _PLUGIN.path
end

--- Get root directory of efface-magique-lr project
function PluginUtils.getProjectRoot()
    local pluginDir = PluginUtils.getPluginDir()
    -- plugin is in <root>/plugin/ai_eraser.lrplugin
    local pluginParent = LrPathUtils.parent(pluginDir)
    local projectRoot = LrPathUtils.parent(pluginParent)
    return projectRoot
end

--- Locate Python executable in .venv or system PATH
function PluginUtils.getPythonExecutable()
    local root = PluginUtils.getProjectRoot()
    local isWindows = (WIN_ENV == true)

    local candidatePaths = {}

    if isWindows then
        table.insert(candidatePaths, LrPathUtils.child(root, ".venv\\Scripts\\python.exe"))
        table.insert(candidatePaths, LrPathUtils.child(root, "venv\\Scripts\\python.exe"))
        table.insert(candidatePaths, LrPathUtils.child(PluginUtils.getPluginDir(), ".venv\\Scripts\\python.exe"))
    else
        table.insert(candidatePaths, LrPathUtils.child(root, ".venv/bin/python"))
        table.insert(candidatePaths, LrPathUtils.child(root, "venv/bin/python"))
        table.insert(candidatePaths, LrPathUtils.child(PluginUtils.getPluginDir(), ".venv/bin/python"))
        table.insert(candidatePaths, "/usr/local/bin/python3")
        table.insert(candidatePaths, "/opt/homebrew/bin/python3")
        table.insert(candidatePaths, "/usr/bin/python3")
    end

    for _, path in ipairs(candidatePaths) do
        if LrFileUtils.exists(path) == "file" then
            return path
        end
    end

    -- Default fallback to PATH invocation
    return isWindows and "python" or "python3"
end

--- Locate companion app entrypoint
function PluginUtils.getCompanionAppPath()
    local root = PluginUtils.getProjectRoot()
    local companionScript = LrPathUtils.child(root, "companion" .. (WIN_ENV and "\\app.py" or "/app.py"))
    return companionScript
end

--- Quote a command line argument
function PluginUtils.quoteArg(arg)
    if WIN_ENV then
        return '"' .. tostring(arg):gsub('"', '\\"') .. '"'
    else
        return "'" .. tostring(arg):gsub("'", "'\\''") .. "'"
    end
end

--- Get live bridge port from .tmp/live_bridge.json or default 51739
function PluginUtils.getLiveBridgePort()
    local root = PluginUtils.getProjectRoot()
    local infoPath = LrPathUtils.child(root, ".tmp" .. (WIN_ENV and "\\live_bridge.json" or "/live_bridge.json"))
    if LrFileUtils.exists(infoPath) == "file" then
        local content = LrFileUtils.readFile(infoPath)
        if content then
            local port = content:match('"port"%s*:%s*(%d+)')
            if port then
                return tonumber(port)
            end
        end
    end
    return 51739
end

--- Launch a detached background process so Lightroom UI is never blocked
function PluginUtils.launchBackgroundProcess(command)
    if WIN_ENV then
        local winCmd = string.format('cmd.exe /c start "" %s', command)
        LrTasks.execute(winCmd)
    else
        local posixCmd = string.format('nohup %s >/dev/null 2>&1 &', command)
        LrTasks.execute(posixCmd)
    end
end

return PluginUtils
