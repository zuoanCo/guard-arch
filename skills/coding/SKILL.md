---
name: coding
description: 编码能力：阅读、修改、调试工作区内的代码
tools:
  - read_file
  - write_file
  - edit_file
  - list_directory
  - search_text
  - run_command
  - remember
---

# Coding Skill

工作方式：

1. **先读后写**：修改任何文件之前，先用 read_file / search_text 理解现状。
2. **最小改动**：只改必要的部分，保持项目现有代码风格与约定。
3. **验证**：修改完成后，运行项目已有的测试或构建命令验证；不确定命令时先查看
   README / package.json / pyproject.toml，或用 remember 回忆 project memory。
4. **错误处理**：测试失败时，读取报错、定位原因、修复后重跑，不要盲目重试。
5. 学到的项目事实（例如“本项目用 pnpm test”）用 remember 写入 project memory。
