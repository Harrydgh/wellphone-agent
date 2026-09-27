# Wellphone Agent

让用户继续使用 Android 主屏幕的同时，Agent 在同一台手机的独立虚拟显示中完成任务。

## 当前状态

- [x] Mi 10 Ultra / Android 13 无线 ADB 连接
- [x] scrcpy 独立虚拟显示验证
- [x] 主屏幕与虚拟屏可同时操作不同 App
- [x] 通过 scrcpy 控制通道向虚拟显示发送输入
- [x] 固定浏览器启动与虚拟显示归属验证
- [x] 虚拟屏画面采集与基础页面状态
- [x] UI Automator 结构化文字与控件提取
- [x] scrcpy 二进制控制通道与输入动作封装
- [x] 控制前亮屏、解锁安全检查
- [x] 在已解锁真机上完成点击、返回与滑动验收
- [x] Agent 观察—规划—执行—验证核心循环
- [x] 规则规划器与最大步数、无变化停止保护
- [x] 自然语言安全目标解析
- [x] OpenAI Structured Outputs 模型规划器
- [x] LangGraph 状态工作流与运行内检查点
- [x] LangChain DeepSeek 结构化动作规划器
- [x] DeepSeek `deepseek-flash` 独立 API 验收
- [ ] 截图 OCR（作为 UI Automator 无法读取时的补充）
- [ ] AI 模型规划器真机端到端验收（等待 API 可用额度）
- [x] LangGraph + DeepSeek 真机端到端验收

已知限制：Android App 默认不一定支持多实例。当用户与 Agent 同时打开同一个 App 时，系统可能复用或移动现有任务，导致其中一块显示停在最后一帧。第一版要求双方使用不同 App。

## 第一阶段结论

```text
手机主显示 display 0        Agent 虚拟显示 display N
用户正常操作                scrcpy 创建并显示
键盘和焦点归用户            画面采集 + 页面状态识别
          \                /
           同一台 Android 手机
```

## 环境诊断

保持手机和电脑连接同一 Wi-Fi，并在手机开发者选项中开启“无线调试”：

```powershell
.\scripts\doctor.ps1
```

程序会自动定位 Winget 安装的 ADB、发现已配对手机、连接当前无线端口，并输出设备与浏览器信息。

临时创建虚拟显示并在 10 秒后安全关闭：

```powershell
.\scripts\display_test.ps1
```

运行固定浏览器演示（请不要同时在主屏幕打开同一个浏览器）：

```powershell
.\scripts\run_demo.ps1
```

该流程会在虚拟显示打开 `https://example.com`、确认浏览器任务仍属于虚拟显示，并将事件写入 `logs/`。当前版本不执行支付、登录、删除或权限修改。

运行不连接手机的安全测试：

```powershell
.\scripts\test.ps1
```

连续采集两次虚拟屏画面并输出页面状态 JSON：

```powershell
.\scripts\observe.ps1
```

页面状态包含当前 App、Activity、截图路径、尺寸、内容哈希、视觉变化比例、连续静止帧数、卡帧标记，以及 UI Automator 提取的可见文字、控件资源 ID、边界和可点击状态。调试图片保存在 `screenshots/`，不会提交到 Git。

已确认的设备差异：在本机 Mi 10 Ultra / MIUI 14 上，`adb shell input -d` 虽然返回成功，但不会实际控制 scrcpy 创建的虚拟显示。项目现已改用 scrcpy 控制通道，画面采集和页面状态识别不受此差异影响。

## 第二阶段：虚拟屏控制（已完成）

本阶段已接入与本机 scrcpy 版本匹配的控制服务，支持在独立虚拟屏中启动 App、点击、滑动、返回和安全文本输入。所有坐标输入只通过绑定虚拟显示的控制连接发送；缺少连接时会拒绝执行，不会回退到主屏输入。

真机诊断已确认：MIUI 在手机息屏或锁屏时仍会接收控制消息，但会丢弃发往虚拟屏的点击和按键。因此控制测试现在会在启动前检查手机状态，并在未点亮或未解锁时停止并提示。

请先点亮并解锁手机，然后运行：

```powershell
.\scripts\control_test.ps1
```

测试会在临时虚拟屏打开系统设置，自动识别一个安全的设置行，依次验证点击、返回和滑动，并把截图与动作日志分别保存到 `screenshots/` 和 `logs/`。测试不会修改设置开关。

2026-09-25 已在 Mi 10 Ultra / Android 13 / MIUI 14 上完成实机验收：虚拟显示 36 中点击 `WLAN` 后进入 `.Settings$WifiSettingsActivity`，返回后恢复 `.MainSettings`，随后滑动使可见设置项和画面哈希发生变化。三项动作均通过 Activity、UI Automator 页面文字和截图变化交叉确认。

## 第三阶段：Agent 核心循环（已完成）

本阶段将页面感知和虚拟屏控制连接成目标驱动循环：Agent 每一步先观察当前页面，再由规划器选择动作，经过安全策略批准后执行，随后重新观察页面并验证结果。当前首个受控目标是“打开系统 WLAN 设置页面”。

```text
任务目标 → 观察页面 → 规划动作 → 安全检查 → 执行动作
              ↑                         ↓
              └──── 验证变化 / 重试 / 停止 ────┘
```

当前规则规划器只能生成点击目标、浏览列表、返回、完成和中止，不接收任意坐标。点击坐标必须从当前 UI Automator 页面中的可点击控件重新计算；观察显示与控制显示不一致、离开允许的 App、虚假完成、连续两次无页面变化或超过最大步数时都会停止。

保持手机亮屏并解锁后运行首个 Agent 任务：

```powershell
.\scripts\agent_test.ps1 -Target WLAN
```

2026-09-26 已完成真机验收：Agent 在虚拟屏中观察系统设置，自主选择并点击 `WLAN`，用 1 个动作进入 `.Settings$WifiSettingsActivity`，随后根据目标成功条件自动结束。每次观察、决策、动作和验证均写入 `logs/`，对应截图保存在 `screenshots/`。

这一阶段使用确定性的规则规划器验证 Agent 架构和安全边界，还没有调用云端 AI 模型。下一阶段将在不改变执行安全层的前提下，增加自然语言任务解析和模型规划器。

## 第四阶段：AI 模型规划器（代码完成，待实机验收）

本阶段增加自然语言安全目标解析和 OpenAI 模型规划器。用户任务必须先映射成本地白名单目标；模型只能在 `tap`、`scroll_down`、`abort` 三种结构化动作中选择，不能生成坐标、不能扩大 App 范围，也不能自行宣布任务完成。模型建议仍须经过第三阶段的本地安全策略，完成状态仍由手机 Activity 在本地验证。

为减少隐私暴露，模型请求不包含截图、完整页面文字、账号或周边 Wi-Fi 名称，只包含任务名称、当前 App/Activity、目标是否可点击、已浏览次数和允许动作，并显式设置 `store=false`。响应使用严格 JSON Schema；认证、额度和网络错误会转换为不包含凭据内容的本地提示。

使用默认模型运行：

```powershell
.\scripts\ai_agent_test.ps1 -Task "请帮我打开 Wi-Fi 设置"
```

也可以通过 `WELLPHONE_OPENAI_MODEL` 或脚本的 `-Model` 参数选择账户有权限使用的模型。项目使用 `OPENAI_API_KEY`，密钥不得写入代码、README、日志或 Git。

2026-09-26 离线结构化输出、越权目标拒绝、自然语言目标白名单和 API 错误脱敏测试已经通过。真实 API 测试已到达 OpenAI 服务端，但当前备用账户返回额度不足，因此本阶段暂不标记为真机端到端完成；增加可用额度后需重新运行上述命令完成验收。

## 第五阶段：LangGraph 工作流与 DeepSeek（已完成）

本阶段在保留原有 `AgentLoop` 稳定基线的同时，新增 LangGraph 状态工作流。新流程由 `observe`、`plan`、`validate`、`execute`、`verify` 和 `finalize` 六个节点组成，并通过条件边完成成功、失败、重试和安全停止。每次运行使用独立 `thread_id` 和内存检查点，便于追踪同一次任务的状态变化。

DeepSeek 通过 LangChain 官方 `langchain-deepseek` 适配器接入。模型只能返回 `tap`、`scroll_down` 或 `abort`，点击目标仍必须等于本地白名单目标；坐标由 UI Automator 当前页面重新计算。模型不能直接调用 ADB、scrcpy 或任意坐标，所有动作仍须经过现有 `AgentSafetyPolicy`。

模型请求只包含任务、App、Activity、目标是否可点击和安全浏览次数，不上传截图、设备标识、完整页面文字或周边 WLAN 名称。API 错误不会把密钥或服务端敏感详情写入日志。

配置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
$env:WELLPHONE_DEEPSEEK_MODEL = "deepseek-flash"
```

保持手机亮屏、解锁并在线后运行：

```powershell
.\scripts\langgraph_agent_test.ps1 -Task "请帮我打开 Wi-Fi 设置"
```

2026-09-27 已使用 `deepseek-flash` 完成独立 API 和真机端到端验收：LangGraph 在虚拟显示中观察系统设置，DeepSeek 返回受限制的 `tap WLAN` 动作，本地安全策略批准并执行 1 次点击，最终由本地 Activity 验证进入 `.Settings$WifiSettingsActivity`。对应截图和 JSONL 日志保存在 `screenshots/` 与 `logs/`。DeepSeek 偶发格式差异会在执行动作前进行一次安全重试，并由本地 Pydantic 严格校验；两次均失败则停止，不会点击手机。项目共 42 项自动测试通过。

## 安全边界

- 默认不向主显示 `display 0` 发送输入。
- 自动化动作必须通过绑定虚拟显示的 scrcpy 控制连接发送。
- 第一版不执行支付、删除、账号或权限变更。
- 每次运行都应保留动作日志，且提供立即停止方式。
