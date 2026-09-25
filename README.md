# Wellphone Agent

让用户继续使用 Android 主屏幕的同时，Agent 在同一台手机的独立虚拟显示中完成任务。

## 当前状态

- [x] Mi 10 Ultra / Android 13 无线 ADB 连接
- [x] scrcpy 独立虚拟显示验证
- [x] 主屏幕与虚拟屏可同时操作不同 App
- [x] 输入事件可定向发送到虚拟显示
- [ ] 固定浏览器自动化流程
- [ ] 页面观察与结果验证
- [ ] AI Agent 决策循环

已知限制：Android App 默认不一定支持多实例。当用户与 Agent 同时打开同一个 App 时，系统可能复用或移动现有任务，导致其中一块显示停在最后一帧。第一版要求双方使用不同 App。

## 第一阶段结论

```text
手机主显示 display 0        Agent 虚拟显示 display N
用户正常操作                scrcpy 创建并显示
键盘和焦点归用户            ADB 输入定向到 display N
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

该流程会在虚拟显示打开 `https://example.com`、执行一次滚动、确认浏览器任务仍属于虚拟显示，并将事件写入 `logs/`。当前版本不执行支付、登录、删除或权限修改。

运行不连接手机的安全测试：

```powershell
.\scripts\test.ps1
```

## 安全边界

- 默认不向主显示 `display 0` 发送输入。
- 自动化动作必须显式指定虚拟显示编号。
- 第一版不执行支付、删除、账号或权限变更。
- 每次运行都应保留动作日志，且提供立即停止方式。
