临时公告板（贡献与协作规范）：
1. 如果不熟悉git使用，请找组长，不要乱上传Github，会被组长达斯(好吧并不会，上手确实不容易，组长也最多只会站在桌子上生闷气)
2. 尽量保持文件目录整齐，不要加多余的东西 —— 测试脚本和中间产物（尤其是切割后的图片）请务必加入".gitignore"中，不要混入源代码上传喵
3. 文件名和提交记录必须使用英文；代码注释建议用中英双语，最后提交时删掉中文的（英语好的当我没说XoX）
4. 由于团队内部开发环境多元，请勿直接将 pip freeze 的全部结果覆盖到 requirements.txt 中。
5. 如需引入新的第三方 Python 包，请手动将其追加至 requirements.txt 并注明版本区间。


项目名称 (Project Title)
    GrepL，校园失物招领智能匹配系统：基于轻量 NLP 查询重构 + CLIP 图文检索 + 时空辅助排序

环境要求
    1. 基础 Python 环境
    请确保已安装 Python=3.10 并可以正常使用

    2. 核心 Python 依赖（跨平台通用）
    除深度学习计算框架外，项目所需的通用 Python 库已记录在 `requirements.txt` 中。请先执行以下命令安装：
        pip install -r requirements.txt

深度学习框架安装指南 (TensorFlow & TF-Text)
    请根据你本机的操作系统和硬件条件，选择以下其中一种方式安装 TensorFlow。

    选项 A：Windows / Linux (具有 NVIDIA GPU 且需硬件加速)
        如果你需要使用 GPU 进行模型推理或训练（注意我们没有人强制需要！），请确保本地已安装适应你GPU型号的 CUDA 12+版本 和 cuDNN 8.9+（请自行核对 TF 2.18 的标准要求）。
        执行以下命令安装支持 GPU 的 TensorFlow 及相关组件：
            - 安装包含 CUDA 运行时的 TensorFlow
                pip install tensorflow[and-cuda]>=2.18.0
            - 安装对应的文本处理组件
                pip install tensorflow-text>=2.18.0

    选项 B：Windows / Linux / Mac (仅使用 CPU 推理)
    如果你的机器没有 NVIDIA 显卡，或者你只需要运行模型完成简单的推理任务，直接安装 CPU 版本即可，无需配置 CUDA 驱动：
        pip install tensorflow-cpu>=2.18.0
        pip install tensorflow-text>=2.18.0

    选项 C：Mac (Apple Silicon - M1/M2/M3/M4 芯片)
    Mac 芯片用户需要使用以下命令以获得由 Apple Metal API 提供的硬件加速：
        pip install tensorflow>=2.18.0
        pip install tensorflow-text>=2.18.0 (注：从 TF 2.16 开始，Mac 平台的加速已原生集成在主包中，无需额外安装 tensorflow-metal，但请确保 macOS 系统版本较新)

快速开始
    配置环境：参考上述指南完成依赖安装。
    运行模型：执行以下脚本运行推理或应用。
        python main.py

浏览器界面说明
    当前项目提供 NiceGUI 浏览器界面，运行后会打开本地网页：
        python main.py

    如果提示 NiceGUI 未安装，请先执行：
        pip install -r requirements.txt

查询重构说明
    用户输入的英文自然语言描述会先经过 `src/query_refiner.py` 清洗，去除时间、
    地点、不确定语气等非视觉信息，并尽量重写为适合 CLIP 检索的英文短文本。

    示例：
        I probably lost a blue bottle with stickers near the library yesterday
        -> blue water bottle with stickers

    如果系统无法稳定抽取颜色、物品类别或外观特征，会保留原始描述作为 fallback，
    避免误删用户提供的有效信息。
