临时公告板（贡献与协作规范）：
1. 如果不熟悉git使用，请找组长，不要乱上传Github，会被组长达斯(好吧并不会，上手确实不容易，组长也最多只会站在桌子上生闷气)
2. 尽量保持文件目录整齐，不要加多余的东西 —— 测试脚本和中间产物（尤其是切割后的图片）请务必加入".gitignore"中，不要混入源代码上传喵
3. 文件名和提交记录必须使用英文；代码注释建议用中英双语，最后提交时删掉中文的（英语好的当我没说XoX）
4. 由于团队内部开发环境多元，请勿直接将 pip freeze 的全部结果覆盖到 requirements.txt 中。
5. 如需引入新的第三方 Python 包，请手动将其追加至 requirements.txt 并注明版本区间。


项目名称 (Project Title)
    GrepL，一个基于Tensorflow架构的AI驱动失物招领系统

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
        pip install tensorflow-text>=2.18.0 (可选：Windows 上可能没有对应版本的 pip wheel；本项目的 CLIP 推理不强依赖该包)

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
    或：
        python src/main.py

    如果提示 NiceGUI 未安装，请先执行：
        pip install -r requirements.txt


References & Attribution
    项目灵感与接口设计参考（未直接复制外部代码，如有后续引用外部代码片段，会在对应文件中用注释注明来源 URL）：
    - Project idea lists:
      https://www.upgrad.com/blog/artificial-intelligence-projects-in-python/
      https://careerkarma.com/blog/python-projects-beginners/
    - CLIP / KerasHub documentation:
      https://keras.io/api/keras_hub/models/clip/
      https://github.com/keras-team/keras-hub
    - NiceGUI documentation:
      https://nicegui.io/documentation
