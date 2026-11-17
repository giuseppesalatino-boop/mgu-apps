# BMW iDrive 7 MGU 应用合集

本项目由 [giuseppesalatino-boop](https://github.com/giuseppesalatino-boop) 维护。三个应用都可以从 MGU Store 单独安装和更新。

| 应用 | 功能 |
| --- | --- |
| [精准计时（True Time）](true-time) | 记录加速与制动测试，利用 DSC 加速度信号修正道路坡度对结果的影响。 |
| [ECU 实时数据（ECU Live Data）](ecu-live-data) | 只读采集柴油 DDE 或汽油 DME 的数据，支持 CSV 记录、道路测功和加速计时。基于 [Boostline](https://github.com/clucraft/boostline-idrive)。 |
| [DPF 监测（DPF Monitoring）](dpf-monitoring) | 只读显示柴油颗粒捕集器的积炭、灰分、压差、排气温度和再生记录。可用项目取决于车辆的 DDE。 |

## 安装

- 上游版可在 [MGU Store](https://github.com/robertnisipeanu/mgu-store) 的手机页面添加 GitHub 源 `giuseppesalatino-boop/mgu-apps`，选择 **Tags** 或 **main**。
- 简体中文构建使用独立的应用 ID `true_time_cn`、`boostline_cn` 和 `dpf_id7_cn`，适用于 **iDrive 7**。在小白中文源中按上表中文名称查找。需要支持第三方应用包和车机 SDK 补全的 MGU Store／MGU Deploy。

## 构建简体中文版

```sh
python tools/build_unit.py true_time_cn true-time/app --language zh-CN --without-sdk
python tools/build_unit.py boostline_cn ecu-live-data/app --language zh-CN --without-sdk
python tools/build_unit.py dpf_id7_cn dpf-monitoring/app --language zh-CN --without-sdk
```

输出在 `vendor/idrive7-apps/dist/`。安装时由 MGU Deploy 使用本机兼容的 SDK。中文词条使用内置的本地化目录，涵盖 RHMI 菜单和 GPU 界面，诊断标识、记录格式和单位保留原格式。

加速、制动和道路测功请在封闭场地进行。

## 致谢

- 原作者及项目维护者为 giuseppesalatino-boop。共享工具包由 cauliflowermossgrape 开发，ECU 采集核心来自 Boostline。
- 简体中文翻译由抖音「小白科技杂物社」／BiliBili「小白151」／GitHub `JackieZ123430` 提供。
- 原项目和各上游组件的版权及许可见 [LICENSE](LICENSE) 和各目录的许可证。
