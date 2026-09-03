--[[----------------------------------------------------------------------------
Info.lua
Efface Magique LR - AI Generative Eraser for Adobe Lightroom Classic
Plugin specification and metadata
------------------------------------------------------------------------------]]

return {
    LrSdkVersion = 11.0,
    LrSdkMinimumVersion = 11.0,
    LrToolkitIdentifier = 'com.effacemagique.lightroom',
    LrPluginName = "Efface Magique - AI Generative Eraser",
    LrPluginInfoUrl = "https://github.com/efface-magique-lr",

    -- File > Plug-in Extras (available across Library, Develop, and all modules)
    LrExportMenuItems = {
        {
            title = "⚡ AI Generative Eraser (Live Window)...",
            file = "LiveBridge.lua",
        },
        {
            title = "🪄 AI Generative Eraser (Single Photo)...",
            file = "GenerativeEraser.lua",
        },
    },

    -- Library module top menu & right-click photo context menu
    LrLibraryMenuItems = {
        {
            title = "⚡ AI Generative Eraser (Live Window)...",
            file = "LiveBridge.lua",
        },
        {
            title = "🪄 AI Generative Eraser (Single Photo)...",
            file = "GenerativeEraser.lua",
        },
    },

    -- Help menu entry
    LrHelpMenuItems = {
        {
            title = "About AI Generative Eraser...",
            file = "GenerativeEraser.lua",
        },
    },

    LrPluginProperties = {
        includeSubfolders = true,
    },
}

