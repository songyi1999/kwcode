"""
ReAct Loop: 多轮交互式代码修复Agent。

LLM在环境中自主决定：读文件、写文件、跑测试、搜索代码，
每步看到真实反馈，直到提交最终方案或达到步数上限。

替代 _run_targeted_fix 中的单次LLM调用模式，
让模型能够迭代式探索和修复，突破单次黑盒调用的天花板。
"""

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Tool Definitions (给LLM看的工具描述) ──

TOOL_DESCRIPTIONS = """\
你可以使用以下工具来探索和修改代码。每次回复只能调用一个工具。

## 可用工具

### read_file
读取文件内容。
格式: <tool>read_file</tool><arg>文件路径</arg>

### write_file
写入完整文件内容（覆盖）。
格式: <tool>write_file</tool><arg>文件路径</arg>
<content>
完整文件内容
</content>

### run_test
运行项目测试，查看哪些通过哪些失败。
格式: <tool>run_test</tool>

### grep
在项目中搜索代码模式。
格式: <tool>grep</tool><arg>搜索模式</arg>

### list_dir
列出目录内容。
格式: <tool>list_dir</tool><arg>目录路径（可选，默认项目根目录）</arg>

### submit
提交最终方案，结束修复。确认所有修改已完成后调用。
格式: <tool>submit</tool>

## 重要规则
- 每次回复只调用一个工具
- 先读文件了解现状，再修改
- 修改后跑测试验证
- 不要修改测试文件
- 目标：让所有测试通过
"""


@dataclass
class ReactStep:
    """一步ReAct交互的记录。"""
    thought: str = ""
    tool: str = ""
    tool_arg: str = ""
    tool_content: str = ""  # write_file的content
    observation: str = ""
    elapsed_ms: float = 0


@dataclass
class ReactResult:
    """ReAct循环的最终结果。"""
    success: bool = False
    steps: list = field(default_factory=list)
    final_files: dict = field(default_factory=dict)  # {path: content}
    tests_passed: int = 0
    tests_total: int = 0
    total_elapsed_s: float = 0


class ReactLoop:
    """
    多轮ReAct循环Agent。

    使用方式:
        loop = ReactLoop(llm, tools, max_steps=10)
        result = loop.run(ctx, target_files)
    """

    def __init__(self, llm, tools, max_steps: int = 10, step_timeout: int = 120):
        """
        Args:
            llm: LLMBackend instance (支持 chat/generate)
            tools: ToolExecutor instance
            max_steps: 最大交互步数
            step_timeout: 每步工具执行超时(秒)
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.step_timeout = step_timeout

    def run(self, ctx, target_files: list[str]) -> ReactResult:
        """
        执行ReAct循环。

        Args:
            ctx: TaskContext (包含 user_input, project_root, verifier_output 等)
            target_files: 需要修复的目标文件列表

        Returns:
            ReactResult with final state
        """
        t0 = time.time()
        result = ReactResult()

        # 构建初始system prompt
        system = self._build_system_prompt(ctx, target_files)

        # 构建初始user message（任务描述+当前状态）
        initial_msg = self._build_initial_message(ctx, target_files)

        # 对话历史
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": initial_msg},
        ]

        # 记录文件修改（用于最终输出）
        modified_files = {}

        for step_idx in range(self.max_steps):
            step = ReactStep()
            step_t0 = time.time()

            # 调用LLM
            # 根据模型大小调整token预算
            is_small = self._is_small_model()
            max_tokens = 2048 if is_small else 4096

            response = self.llm.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            if not response:
                logger.warning("[react] LLM returned empty response at step %d", step_idx)
                break

            # 解析LLM输出：thought + tool call
            step.thought, step.tool, step.tool_arg, step.tool_content = self._parse_response(response)

            logger.info("[react] step %d: tool=%s arg=%s", step_idx, step.tool, step.tool_arg[:80] if step.tool_arg else "")

            # 执行工具
            if step.tool == "submit":
                step.observation = "已提交。"
                step.elapsed_ms = (time.time() - step_t0) * 1000
                result.steps.append(step)
                result.success = True
                break

            elif step.tool == "read_file":
                step.observation = self._exec_read_file(ctx, step.tool_arg)

            elif step.tool == "write_file":
                path = step.tool_arg
                content = step.tool_content
                ok = self._exec_write_file(ctx, path, content)
                if ok:
                    modified_files[path] = content
                    step.observation = f"已写入 {path} ({len(content)} bytes)"
                else:
                    step.observation = f"写入失败: {path}"

            elif step.tool == "run_test":
                test_result = self._exec_run_test(ctx)
                step.observation = test_result
                # 解析测试结果
                passed, total = self._parse_test_counts(test_result)
                result.tests_passed = passed
                result.tests_total = total

            elif step.tool == "grep":
                step.observation = self._exec_grep(ctx, step.tool_arg)

            elif step.tool == "list_dir":
                step.observation = self._exec_list_dir(ctx, step.tool_arg)

            else:
                # 无法识别的工具或LLM没有调用工具
                step.observation = f"未识别的工具: {step.tool}。请使用可用工具之一。"
                if not step.tool:
                    # LLM可能只输出了思考没有调用工具，提醒它
                    step.observation = "请调用一个工具来继续。可用工具: read_file, write_file, run_test, grep, list_dir, submit"

            step.elapsed_ms = (time.time() - step_t0) * 1000
            result.steps.append(step)

            # 将assistant回复和observation加入对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"[观察结果]\n{step.observation}"})

            # 上下文窗口管理：如果历史太长，压缩早期步骤
            messages = self._maybe_compress_history(messages)

        result.final_files = modified_files
        result.total_elapsed_s = time.time() - t0

        # 如果没有显式submit但有修改，也算部分成功
        if not result.success and modified_files:
            # 跑一次最终测试确认状态
            final_test = self._exec_run_test(ctx)
            passed, total = self._parse_test_counts(final_test)
            result.tests_passed = passed
            result.tests_total = total

        logger.info("[react] completed: %d steps, %d files modified, %d/%d tests, %.1fs",
                    len(result.steps), len(modified_files),
                    result.tests_passed, result.tests_total, result.total_elapsed_s)

        return result

    # ── System & Initial Message Construction ──

    def _build_system_prompt(self, ctx, target_files: list[str]) -> str:
        """构建system prompt，包含工具描述和约束。"""
        parts = [
            "你是一个代码修复Agent。你的任务是通过多轮交互修复代码中的bug，使所有测试通过。",
            "",
            TOOL_DESCRIPTIONS,
            "",
            "## 工作流程建议",
            "1. 先 read_file 查看目标文件和测试文件",
            "2. 分析失败原因",
            "3. write_file 修复代码",
            "4. run_test 验证修复",
            "5. 如果还有失败，继续分析和修复",
            "6. 全部通过后 submit",
            "",
            f"## 项目根目录: {ctx.project_root}",
            f"## 目标文件: {', '.join(target_files)}",
        ]
        # rename/refactor任务追加规则
        user_lower = ctx.user_input.lower()
        if any(kw in user_lower for kw in ["rename", "重命名", "refactor", "重构"]):
            parts.append("")
            parts.append("## 重命名/重构规则")
            parts.append("- 必须用grep找到所有引用点，不能只改目标文件")
            parts.append("- 修改顺序：先改定义，再改所有调用点")
            parts.append("- 全部改完再run_test")
        return "\n".join(parts)

    def _build_initial_message(self, ctx, target_files: list[str]) -> str:
        """构建初始消息，包含任务描述和当前失败信息。"""
        parts = [f"## 任务\n{ctx.user_input}"]

        # 检测rename/refactor任务，注入多文件策略
        user_lower = ctx.user_input.lower()
        rename_kw = ["rename", "重命名", "refactor", "重构", "split", "拆分"]
        if any(kw in user_lower for kw in rename_kw):
            parts.append(
                "\n## 多文件重命名策略\n"
                "1. 先用 grep 搜索旧名称，找到所有包含它的文件\n"
                "2. 逐个 read_file 读取每个相关文件\n"
                "3. 逐个 write_file 修改每个文件中的所有引用\n"
                "4. 最后 run_test 验证\n"
                "不要遗漏任何文件中的引用！"
            )

        # 当前测试失败信息
        if ctx.verifier_output:
            error_detail = ctx.verifier_output.get("error_detail", "")
            if error_detail:
                # 截断过长的错误信息
                if len(error_detail) > 2000:
                    error_detail = error_detail[:2000] + "\n... (截断)"
                parts.append(f"\n## 当前测试失败\n{error_detail}")

            passed = ctx.verifier_output.get("tests_passed", 0)
            total = ctx.verifier_output.get("tests_total", 0)
            if total > 0:
                parts.append(f"\n当前状态: {passed}/{total} 测试通过")

        # 已有的最佳进展
        if ctx.best_tests_passed > 0:
            parts.append(f"\n已有最佳进展: {ctx.best_tests_passed} 个测试通过，请在此基础上继续修复。")

        # retry hint
        if ctx.retry_hint:
            parts.append(f"\n## 提示\n{ctx.retry_hint}")

        parts.append("\n请开始修复。先读取相关文件了解现状。")
        return "\n".join(parts)

    # ── Response Parsing ──

    def _parse_response(self, response: str) -> tuple[str, str, str, str]:
        """
        解析LLM回复，提取thought和tool call。

        Returns: (thought, tool_name, tool_arg, tool_content)
        """
        # 提取tool call
        tool_match = re.search(r'<tool>(.*?)</tool>', response)
        arg_match = re.search(r'<arg>(.*?)</arg>', response, re.DOTALL)
        content_match = re.search(r'<content>\n?(.*?)</content>', response, re.DOTALL)

        tool = tool_match.group(1).strip() if tool_match else ""
        tool_arg = arg_match.group(1).strip() if arg_match else ""
        tool_content = content_match.group(1) if content_match else ""

        # thought是tool标签之前的所有文本
        if tool_match:
            thought = response[:tool_match.start()].strip()
        else:
            thought = response.strip()

        return thought, tool, tool_arg, tool_content

    # ── Tool Execution ──

    def _exec_read_file(self, ctx, path: str) -> str:
        """读取文件，返回内容或错误。"""
        if not path:
            return "[ERROR] 请提供文件路径"
        content = self.tools.read_file(path)
        if content and not content.startswith("[ERROR]"):
            # 截断过大的文件
            lines = content.split('\n')
            if len(lines) > 300:
                return '\n'.join(lines[:300]) + f"\n\n... (文件共{len(lines)}行，已截断前300行)"
        return content

    def _exec_write_file(self, ctx, path: str, content: str) -> bool:
        """写入文件。禁止写测试文件。"""
        if not path or not content:
            return False
        # 禁止修改测试文件
        if "test" in path.lower():
            logger.warning("[react] blocked write to test file: %s", path)
            return False
        # 语法检查（仅Python）
        if path.endswith('.py'):
            try:
                ast.parse(content)
            except SyntaxError as e:
                logger.warning("[react] syntax error in write: %s", e)
                return False
        return self.tools.write_file(path, content)

    def _exec_run_test(self, ctx) -> str:
        """运行测试，返回输出。"""
        from kaiwu.core.context import TaskContext as _TC
        from kaiwu.experts.verifier import VerifierExpert as _VE

        _tmp_ctx = _TC(project_root=ctx.project_root)
        _tmp_ver = _VE(self.llm, self.tools)
        result = _tmp_ver.run_tests_only(_tmp_ctx)

        output = result.get("output", "")
        passed = result.get("passed", 0)
        total = result.get("total", 0)

        # 构建简洁的测试摘要
        summary = f"测试结果: {passed}/{total} 通过"
        if passed == total and total > 0:
            summary += " (全部通过!)"

        # 附加失败详情（截断）
        if output and passed < total:
            if len(output) > 1500:
                output = output[:1500] + "\n... (截断)"
            return f"{summary}\n\n{output}"
        return summary

    def _exec_grep(self, ctx, pattern: str) -> str:
        """在项目中搜索代码。"""
        if not pattern:
            return "[ERROR] 请提供搜索模式"
        import subprocess
        try:
            # 使用grep搜索（跨平台兼容）
            cmd = f'grep -rn "{pattern}" --include="*.py" --include="*.ts" --include="*.js" --include="*.go" .'
            stdout, stderr, rc = self.tools.run_bash(cmd, cwd=ctx.project_root, timeout=10)
            if stdout:
                lines = stdout.strip().split('\n')
                if len(lines) > 30:
                    return '\n'.join(lines[:30]) + f"\n... (共{len(lines)}个匹配，显示前30个)"
                return stdout.strip()
            return f"未找到匹配: {pattern}"
        except Exception as e:
            return f"[ERROR] grep failed: {e}"

    def _exec_list_dir(self, ctx, path: str) -> str:
        """列出目录内容。"""
        target = path if path else "."
        entries = self.tools.list_dir(target)
        if isinstance(entries, list):
            if entries and entries[0].startswith("[ERROR]"):
                return entries[0]
            return '\n'.join(entries[:50])
        return str(entries)

    # ── Helpers ──

    def _is_small_model(self) -> bool:
        """检测是否为小模型。"""
        model_name = getattr(self.llm, 'ollama_model', '').lower()
        return any(s in model_name for s in ('1b', '3b', '4b', '7b', '8b'))

    def _parse_test_counts(self, test_output: str) -> tuple[int, int]:
        """从测试输出中解析通过/总数。"""
        # 匹配 "X/Y 通过" 格式
        m = re.search(r'(\d+)/(\d+)\s*通过', test_output)
        if m:
            return int(m.group(1)), int(m.group(2))
        # 匹配 pytest 格式 "X passed, Y failed"
        passed_m = re.search(r'(\d+)\s*passed', test_output)
        failed_m = re.search(r'(\d+)\s*failed', test_output)
        passed = int(passed_m.group(1)) if passed_m else 0
        failed = int(failed_m.group(1)) if failed_m else 0
        if passed or failed:
            return passed, passed + failed
        return 0, 0

    def _maybe_compress_history(self, messages: list[dict]) -> list[dict]:
        """
        如果对话历史过长，压缩早期步骤。
        保留system + 初始user + 最近6轮交互。
        """
        # system(1) + initial_user(1) + pairs(assistant+user) = 2 + 2*N
        # 保留最近6轮 = 12条消息 + 2条头部 = 14条
        max_messages = 14
        if len(messages) <= max_messages:
            return messages

        # 保留 system + initial_user + 最近的交互
        head = messages[:2]  # system + initial user
        tail = messages[-(max_messages - 2):]  # 最近的交互

        # 插入压缩摘要
        compressed_count = len(messages) - max_messages
        summary = f"[前{compressed_count // 2}步已压缩。你已经读取了文件并进行了一些修改。请继续基于最近的观察结果工作。]"
        head.append({"role": "user", "content": summary})

        return head + tail
