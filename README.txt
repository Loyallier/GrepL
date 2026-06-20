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

数据库说明
    data 文件夹作为当前项目的数据库，负责保存原始图片信息、裁剪后的物品图片、
    可搜索物品记录，以及图像向量化相关文件和索引信息。

    raw_found_images 文件夹：
        存储未经过裁剪的原始图片，就是一批同时被找到物品的合照

    raw_found_image_info.json：
        存储原始图片对应的入库信息，对应 RawFoundItem，每条记录对应一张原始图片
        主要字段包括：
            raw_id：原始图片编号
            image_path：原始图片路径
            found_time：这一批物品被找到的时间
            found_location：这一批物品被找到的位置
            status：原图处理状态，比如 pending
            processed_at：登记中控处理完成后写入的时间
            item_count：这张原图最终生成的 LostItem 数量
            error：仅在处理失败时写入错误信息。

    cropped_item_image 文件夹：
        存储 detector 从原始图片中裁剪出的单个物品图片

    generated 文件夹：
        存储登记、搜索和向量化流程中生成的中间产物和结果文件。

        found_items.json：
            存储已经完成登记、可以被搜索的 LostItem 记录，每条记录对应一个裁剪后的单个物品
            主要字段包括：
                item_id：单个可搜索物品的唯一编号
                raw_id：对应该物品来源于哪一张原始图片
                image_path：裁剪后单个物品图片的路径
                found_time：继承自 RawFoundItem 的 found_time
                found_location：继承自 RawFoundItem 的 found_location
                bound_confidence：检测模型裁剪该物品区域的置信度
                category：可选的单个物品分类，可能是一个多余的属性
                embedding_registered：该物品是否已经完成图像向量登记。
                registered_at：该 LostItem 写入 found_items.json 的时间。

        image_embeddings.json：
            存储图像向量化流程生成的向量数据。
            当前 embedding_engine 会把每个 item_id 对应的图像向量、图片路径、模型信息和更新时间写入该文件。
            该文件是当前版本实际使用的图像向量库。

原图及相关信息后台加入脚本说明
    scripts/add_raw_found_images.py 是一个工具脚本，可以在后台导入原始拾获图片及相关信息的工具脚本。
    这个脚本只负责把原始图片及其对应的发现时间、发现地点等信息加入 data 数据库。

    以下的是使用脚本的基本流程：
        1. 在命令行中运行脚本，并通过参数传入一个本地图片文件夹路径
        2. 脚本会依次读取该文件夹的图片文件
        3. 每张图片会被复制到 data/raw_found_images 文件夹中
        4. 脚本会为每张图片生成新的规范名称和 raw_id，比如 raw_20260621_001
        5. 脚本会在终端中逐张询问一张原图中所有物品共同对应的发现时间（包括日期，小时）和发现地点
        6. 脚本会把 raw_id、原图路径、发现时间、发现地点和 pending 状态绑定成一条记录，并且写入 data/raw_found_image_info.json
        7. 导入完成后，终端会输出本次已经导入的原图数量和下一步提示需要完成的等

    使用的时候终端运行：
        python scripts/add_raw_found_images.py "需要导入的图片文件夹路径" （记得替换路径）

    支持的图片格式：
        .jpg, .jpeg, .png, .webp, .bmp

    这个脚本的注意事项：
        1. 传入的路径必须是一个存在的本地文件夹。
        2. 当前脚本只读取该文件夹第一层中的图片，不会递归读取子文件夹
        3. 原始文件夹中的图片不会被删除，脚本只会复制一份
        4. 每张原图只会生成一条 RawFoundItem 记录；如果一张原图中有多个物品，它们共享同一个发现时间和发现地点
        5. 脚本写入的记录默认 status 为 pending，后续需要再运行 registration_service.register_pending_raw_found_items() 完成检测、裁剪、LostItem 生成和向量登记
