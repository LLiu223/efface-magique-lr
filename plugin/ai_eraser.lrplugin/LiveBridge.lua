--[[----------------------------------------------------------------------------
LiveBridge.lua
Efface Magique LR - Adobe Lightroom Classic Plugin
Seamless Live Sync Bridge: Observes Lightroom photo selection, debounces exports,
syncs active photo to Companion window, and auto-imports completed inpaintings.
------------------------------------------------------------------------------]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrExportSession = import 'LrExportSession'
local LrFileUtils = import 'LrFileUtils'
local LrHttp = import 'LrHttp'
local LrPathUtils = import 'LrPathUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local PluginUtils = require 'PluginUtils'
local JsonUtil = require 'JsonUtil'
local ImportGuard = require 'ImportGuard'

local function runLiveBridge()
    local catalog = LrApplication.activeCatalog()

    local progressScope = LrProgressScope({
        title = "⚡ AI Generative Eraser: Live Sync",
        caption = "Connecting to companion window...",
    })
    progressScope:setCancelable(true)

    local root = PluginUtils.getProjectRoot()
    local tempDir = LrPathUtils.child(root, ".tmp")
    if not LrFileUtils.exists(tempDir) then
        LrFileUtils.createAllDirectories(tempDir)
    end

    local port = PluginUtils.getLiveBridgePort()
    local bridgeInfoFile = LrPathUtils.child(tempDir, "live_bridge.json")
    local pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
    local isConnected = false

    -- Check if companion is already running
    local response = LrHttp.get(pingUrl)
    if response and string.find(response, "EffaceMagique") then
        isConnected = true
    end

    -- If not running, launch companion detached in live mode
    if not isConnected then
        progressScope:setCaption("Launching local AI Companion in live mode...")
        local pythonExe = PluginUtils.getPythonExecutable()
        local companionScript = PluginUtils.getCompanionAppPath()

        if LrFileUtils.exists(companionScript) ~= "file" then
            progressScope:done()
            LrDialogs.message(
                "AI Generative Eraser Error",
                "Could not locate companion app at:\n" .. companionScript,
                "critical"
            )
            return
        end

        local launchCmd = string.format(
            '%s %s --live',
            PluginUtils.quoteArg(pythonExe),
            PluginUtils.quoteArg(companionScript)
        )
        PluginUtils.launchBackgroundProcess(launchCmd)

        -- Poll until companion responds to ping
        local attempts = 0
        while not isConnected and attempts < 25 and not progressScope:isCanceled() do
            LrTasks.sleep(0.4)
            attempts = attempts + 1
            port = PluginUtils.getLiveBridgePort()
            pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
            response = LrHttp.get(pingUrl)
            if response and string.find(response, "EffaceMagique") then
                isConnected = true
                break
            end
        end
    end

    if not isConnected then
        progressScope:done()
        if not progressScope:isCanceled() then
            LrDialogs.message(
                "AI Generative Eraser",
                "Could not connect to Live Companion window.",
                "warning"
            )
        end
        return
    end

    progressScope:setPortionComplete(1.0, 1.0)
    progressScope:setCaption("Live Sync Active. Navigating photos updates companion.")

    -- Focus companion window on connect
    local focusUrl = string.format("http://127.0.0.1:%d/api/focus", port)
    LrHttp.post(focusUrl, "{}", { { field = "Content-Type", value = "application/json" } })

    -- Main selection & import polling loop
    local lastPhotoId = nil
    local pendingPhoto = nil
    local pendingTime = 0
    local disconnectedCount = 0
    local DEBOUNCE_DELAY = 0.08 -- 80ms snappy debounce
    local tick = 0 -- pending imports are polled every 10th 30ms tick (~300ms)

    -- Recency list of live_<id>.tif paths exported this session (oldest first); only these are pruned.
    local MAX_CACHED_EXPORTS = 5
    local cachedExports = {}
    local exportEditTimes = {} -- photo id -> lastEditTime at export (false when the photo has none)
    local sessionPhotos = {} -- tostring(photo id) -> original path, for ids sent to /api/select this session

    local function touchCachedExport(path)
        for i = #cachedExports, 1, -1 do
            if cachedExports[i] == path then
                table.remove(cachedExports, i)
            end
        end
        table.insert(cachedExports, path)
        while #cachedExports > MAX_CACHED_EXPORTS do
            LrFileUtils.delete(table.remove(cachedExports, 1))
        end
    end

    local headers = {
        { field = "Content-Type", value = "application/json" }
    }

    while not progressScope:isCanceled() do
        LrTasks.sleep(0.03) -- 30ms high-frequency polling
        tick = tick + 1

        -- Fast close detection: if the info file was deleted by the companion on close, exit immediately
        if LrFileUtils.exists(bridgeInfoFile) ~= "file" then
            break
        end

        local targetPhoto = catalog:getTargetPhoto()
        local targetId = targetPhoto and targetPhoto.localIdentifier or nil

        -- 1. Check selection change with snappy debounce
        if targetId ~= lastPhotoId then
            if targetId ~= (pendingPhoto and pendingPhoto.localIdentifier) then
                pendingPhoto = targetPhoto
                pendingTime = os.clock()
            else
                if (os.clock() - pendingTime) >= DEBOUNCE_DELAY then
                    lastPhotoId = targetId
                    local activePhoto = pendingPhoto
                    pendingPhoto = nil

                    if activePhoto then
                        local origPath = activePhoto:getRawMetadata('path')
                        local baseName = LrPathUtils.removeExtension(LrPathUtils.leafName(origPath))
                        local exportName = string.format("live_%s.tif", tostring(targetId))
                        local exportPath = LrPathUtils.child(tempDir, exportName)

                        local selectUrl = string.format("http://127.0.0.1:%d/api/select", port)
                        local jsonBody = string.format(
                            '{"path": "%s", "photo_id": "%s", "original_path": "%s", "title": "%s"}',
                            JsonUtil.jsonEscape(exportPath), JsonUtil.jsonEscape(targetId), JsonUtil.jsonEscape(origPath), JsonUtil.jsonEscape(baseName)
                        )

                        -- false stands in for a missing edit time so the table entry still exists
                        local editTime = activePhoto:getRawMetadata('lastEditTime') or false
                        local cachedEditTime = exportEditTimes[targetId]

                        -- Cache hit only for files exported this session from the same edit state
                        if LrFileUtils.exists(exportPath) == "file"
                            and cachedEditTime ~= nil and cachedEditTime == editTime then
                            touchCachedExport(exportPath)
                            sessionPhotos[tostring(targetId)] = origPath
                            LrHttp.post(selectUrl, jsonBody, headers)
                        else
                            -- Ultra-fast uncompressed 16-bit TIFF export (no CPU-heavy ZIP compression)
                            local exportSettings = {
                                LR_export_destinationType = 'specificFolder',
                                LR_export_destinationPathPrefix = tempDir,
                                LR_export_useSubfolder = false,
                                LR_collisionHandling = 'overwrite',
                                LR_format = 'TIFF',
                                LR_export_bitDepth = 16,
                                LR_export_colorSpace = 'ProPhotoRGB',
                                LR_export_compressionMethod = 'None',
                                LR_size_doConstrain = false,
                                LR_outputSharpeningOn = false,
                                LR_metadata_exportExif = true,
                                LR_reimportExportedPhoto = false,
                                LR_renamingTokensOn = true,
                                LR_tokens = string.format("live_%s", tostring(targetId)),
                            }

                            local exportSession = LrExportSession({
                                photosToExport = { activePhoto },
                                exportSettings = exportSettings,
                            })

                            exportSession:doExportOnCurrentTask()

                            if LrFileUtils.exists(exportPath) == "file" then
                                exportEditTimes[targetId] = editTime
                                touchCachedExport(exportPath)
                                sessionPhotos[tostring(targetId)] = origPath
                                LrHttp.post(selectUrl, jsonBody, headers)
                            end
                        end
                    end
                end
            end
        end

        -- 2. Check for completed edits waiting to be imported into Lightroom (every 10th tick)
        local isPollTick = tick % 10 == 0
        local importsResp = isPollTick
            and LrHttp.get(string.format("http://127.0.0.1:%d/api/pending_imports", port))
        if not isPollTick then
            -- Selection-only tick
        elseif importsResp then
            disconnectedCount = 0
            if string.find(importsResp, '"imports"') then
                local items = JsonUtil.parsePendingImports(importsResp)
                for _, item in ipairs(items) do
                    -- Only <tempDir>/live_<id>.tif for an id exported this session; origPath is our own record
                    local editedPath, origPath = ImportGuard.validate(item, tempDir, sessionPhotos)
                    local acknowledge = false
                    if not editedPath then
                        local reason = origPath
                        editedPath = item.path
                        progressScope:setCaption("Import skipped (" .. reason .. "): " .. LrPathUtils.leafName(editedPath))
                        acknowledge = true
                    elseif LrFileUtils.exists(editedPath) == "file" then
                        acknowledge = true
                        local ok = LrTasks.pcall(function()
                            local targetForImport = catalog:findPhotoByPath(origPath) or targetPhoto

                            -- Copy edited TIFF adjacent to original photo folder
                            local finalDestPath = editedPath

                            if origPath then
                                local origFolder = LrPathUtils.parent(origPath)
                                local leaf = LrPathUtils.leafName(editedPath)
                                finalDestPath = LrPathUtils.child(origFolder, leaf)

                                if LrFileUtils.exists(finalDestPath) then
                                    LrFileUtils.delete(finalDestPath)
                                end
                                if not LrFileUtils.copy(editedPath, finalDestPath) then
                                    finalDestPath = editedPath
                                end
                            end

                            local imported = false
                            catalog:withWriteAccessDo("Import AI Generative Eraser Edit", function()
                                local newPhoto = catalog:addPhoto(finalDestPath, targetForImport, 'above')
                                if newPhoto then
                                    imported = true
                                    catalog:setSelectedPhotos(newPhoto, { newPhoto })
                                end
                            end)
                            if not imported then
                                error("addPhoto failed")
                            end

                            -- Only now is the AI edit safe in the catalog; drop the temp copy (never the imported file)
                            if finalDestPath ~= editedPath then
                                LrFileUtils.delete(editedPath)
                            end
                        end)

                        if ok then
                            if LrDialogs.showBeep then
                                LrDialogs.showBeep()
                            end
                        else
                            progressScope:setCaption("Import failed: " .. LrPathUtils.leafName(editedPath))
                        end
                    end

                    if acknowledge then
                        -- Acknowledge (also on failure or rejection, so the item is not retried every poll)
                        local doneUrl = string.format("http://127.0.0.1:%d/api/import_done", port)
                        local jsonDoneBody = string.format('{"path": "%s"}', JsonUtil.jsonEscape(editedPath))
                        LrHttp.post(doneUrl, jsonDoneBody, headers)
                    end
                end
            end
        else
            disconnectedCount = disconnectedCount + 1
            -- Companion unreachable for 5 failed polls x ~300ms = ~1.5s: re-read port, then ping before giving up
            if disconnectedCount >= 5 then
                port = PluginUtils.getLiveBridgePort()
                local pingUrl = string.format("http://127.0.0.1:%d/api/ping", port)
                local pingResp = LrHttp.get(pingUrl)
                if not pingResp or not string.find(pingResp, "EffaceMagique") then
                    -- Companion window has been closed! Terminate Live Sync cleanly.
                    break
                else
                    disconnectedCount = 0
                end
            end
        end
    end

    -- If user clicked cancel in Lightroom activity center, request companion window to close
    if progressScope:isCanceled() then
        local closeUrl = string.format("http://127.0.0.1:%d/api/close", port)
        LrHttp.post(closeUrl, "{}", headers)
    end

    -- Clean up temporary exported live TIFFs
    if LrFileUtils.exists(tempDir) == "directory" then
        local entries = LrFileUtils.directoryEntries(tempDir)
        if type(entries) == "function" then
            for entry in entries do
                if string.find(entry, "live_") and (string.find(entry, "%.tif") or string.find(entry, "%.tiff")) then
                    LrFileUtils.delete(entry)
                end
            end
        elseif type(entries) == "table" then
            for _, entry in ipairs(entries) do
                if string.find(entry, "live_") and (string.find(entry, "%.tif") or string.find(entry, "%.tiff")) then
                    LrFileUtils.delete(entry)
                end
            end
        end
    end

    progressScope:done()
end

LrTasks.startAsyncTask(runLiveBridge)
