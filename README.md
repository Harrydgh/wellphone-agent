# Wellphone Agent

让用户继续使用 Android 主屏幕的同时，Agent 在同一台手机的独立虚拟显示中完成任务。

## 当前状态

- [x] Mi 10 Ultra / Android 13 无线 ADB 连接
- [x] scrcpy 独立虚拟显示验证
- [x] 主屏幕与虚拟屏可同时操作不同 App
- [ ] 通过 scrcpy 控制通道向虚拟显示发送输入
- [x] 固定浏览器启动与虚拟显示归属验证
- [x] 虚拟屏画面采集与基础页面状态
- [x] UI Automator 结构化文字与控件提取
- [ ] 截图 OCR（作为 UI Automator 无法读取时的补充）
- [ ] AI Agent 决策循环

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

已确认的设备差异：在本机 Mi 10 Ultra / MIUI 14 上，`adb shell input -d` 虽然返回成功，但不会实际控制 scrcpy 创建的虚拟显示。因此触控、滑动和文字输入暂不标记为完成；下一阶段将改用 scrcpy 控制通道。画面采集和页面状态识别不受此限制。

## 安全边界

- 默认不向主显示 `display 0` 发送输入。
- 自动化动作必须通过绑定虚拟显示的 scrcpy 控制连接发送。
- 第一版不执行支付、删除、账号或权限变更。
- 每次运行都应保留动作日志，且提供立即停止方式。
