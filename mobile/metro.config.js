const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// Support WebAssembly for Web compatibility
config.resolver.assetExts.push("wasm");

module.exports = config;
