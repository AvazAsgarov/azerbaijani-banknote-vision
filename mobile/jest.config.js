module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/__tests__/**/*.test.ts"],
  transform: {
    "^.+\\.tsx?$": ["ts-jest", { tsconfig: { esModuleInterop: true } }],
  },
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json"],
};
