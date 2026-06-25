GrepL 使用说明

一、快速开始流程

1. 配置 Python 环境

请确保本机已安装并可以正常使用：

Python 3.10

2. 安装项目通用依赖

项目通用 Python 库已记录在 requirements.txt 中。请先在项目根目录执行：

pip install -r requirements.txt

注意：深度学习计算框架 TensorFlow 和 tensorflow-text 请根据本机系统与硬件情况单独安装，具体方式见后文“TensorFlow 安装说明”。

3. 安装 TensorFlow 与 tensorflow-text

请根据本机操作系统和硬件条件选择以下一种方式安装。

方式 A：Windows / Linux，具有 NVIDIA GPU 且需要硬件加速

如果需要使用 GPU 进行模型推理或训练，请确保本地已安装适配 GPU 型号的 CUDA 12+ 和 cuDNN 8.9+，并自行核对 TensorFlow 2.18 的官方要求。

执行：

pip install tensorflow[and-cuda]>=2.18.0
pip install tensorflow-text>=2.18.0

方式 B：Windows / Linux / Mac，仅使用 CPU 推理

如果机器没有 NVIDIA 显卡，或者只需要运行简单推理任务，可直接安装 CPU 版本：

pip install tensorflow-cpu>=2.18.0
pip install tensorflow-text>=2.18.0

方式 C：Mac，Apple Silicon 芯片，如 M1 / M2 / M3 / M4

Mac 芯片用户可使用以下命令：

pip install tensorflow>=2.18.0
pip install tensorflow-text>=2.18.0

说明：从 TensorFlow 2.16 开始，Mac 平台的加速已原生集成在主包中，无需额外安装 tensorflow-metal，但请确保 macOS 系统版本较新。

4. 启动浏览器界面

当前项目提供 NiceGUI 浏览器界面。运行后会打开本地网页：

python main.py

如果提示 NiceGUI 未安装，请重新执行：

pip install -r requirements.txt

5. 导入原始拾获图片

如需将一批原始拾获图片加入数据库，请使用：

python scripts/add_raw_found_images.py "需要导入的图片文件夹路径"

示例：

python scripts/add_raw_found_images.py "E:\lost_items\raw_images"

支持的图片格式：

.jpg
.jpeg
.png
.webp
.bmp

运行脚本后，终端会逐张询问每张原图对应的发现时间和发现地点。导入完成后，图片会被复制到 data/raw_found_images 文件夹，并在 data/raw_found_image_info.json 中生成对应记录。

6. 执行完整登记流程

导入原始图片后，运行登记中控，完成检测、裁剪、LostItem 生成和向量登记：

python scripts/run_registration.py

7. 暂时跳过图片向量化

如果只想完成原图处理、检测裁剪和 LostItem 生成，暂时不进行图片向量化，可以执行：

python scripts/run_registration.py --skip-embedding

二、常用脚本说明

1. main.py

用于启动项目主程序和 NiceGUI 浏览器界面。

运行方式：

python main.py

2. scripts/add_raw_found_images.py

用于将本地文件夹中的原始拾获图片导入 data 数据库。

该脚本只负责导入原图及其发现时间、发现地点等基本信息，不负责检测、裁剪和向量化。

基本流程：

1. 在命令行中运行脚本，并通过参数传入一个本地图片文件夹路径
2. 脚本读取该文件夹第一层中的图片文件
3. 每张图片被复制到 data/raw_found_images 文件夹
4. 脚本为每张图片生成规范名称和 raw_id，例如 raw_20260621_001
5. 终端逐张询问该原图中所有物品共同对应的发现时间和发现地点
6. 脚本将 raw_id、image_path、found_time、found_location 和 pending 状态绑定为一条记录
7. 记录写入 data/raw_found_image_info.json
8. 导入完成后，终端输出本次导入数量，并提示下一步需要运行登记流程

注意事项：

1. 传入路径必须是一个存在的本地文件夹

2. 当前脚本只读取该文件夹第一层中的图片，不会递归读取子文件夹

3. 原始文件夹中的图片不会被删除，脚本只会复制一份

4. 每张原图只生成一条 RawFoundItem 记录

5. 如果一张原图中有多个物品，它们共享同一个发现时间和发现地点

6. 脚本写入的记录默认 status 为 pending

7. 后续需要运行 registration_service.register_pending_raw_found_items() 或 scripts/run_registration.py 完成检测、裁剪、LostItem 生成和向量登记

8. scripts/run_registration.py

用于启动后端 register 中控，开始处理原图和相关数据。

正常执行完整登记流程：

python scripts/run_registration.py

暂时跳过图片向量化：

python scripts/run_registration.py --skip-embedding

三、项目介绍

项目名称：

GrepL，一个基于 TensorFlow 架构的 AI 驱动失物招领系统

项目定位：

GrepL 用于对拾获物品进行登记、图像处理、向量化和搜索。系统可以将一张包含多个物品的原始拾获图片处理为多个独立的 LostItem 记录，并为后续搜索和匹配提供数据基础。

四、数据库说明

data 文件夹作为当前项目的数据库，负责保存原始图片信息、裁剪后的物品图片、可搜索物品记录，以及图像向量化相关文件和索引信息。

1. data/raw_found_images

用于存储未经裁剪的原始拾获图片。

这些图片通常是一批同时被找到物品的合照。

2. data/raw_found_image_info.json

用于存储原始图片对应的入库信息。

每条记录对应一张原始图片，对应 RawFoundItem。

主要字段包括：

raw_id：原始图片编号
image_path：原始图片路径
found_time：这一批物品被找到的时间
found_location：这一批物品被找到的位置
status：原图处理状态，例如 pending
processed_at：登记中控处理完成后写入的时间
item_count：这张原图最终生成的 LostItem 数量
error：仅在处理失败时写入错误信息

3. data/cropped_item_image

用于存储 detector 从原始图片中裁剪出的单个物品图片。

4. data/generated

用于存储登记、搜索和向量化流程中生成的中间产物和结果文件。

5. data/generated/found_items.json

用于存储已经完成登记、可以被搜索的 LostItem 记录。

每条记录对应一个裁剪后的单个物品。

主要字段包括：

item_id：单个可搜索物品的唯一编号
raw_id：该物品来源于哪一张原始图片
image_path：裁剪后单个物品图片的路径
found_time：继承自 RawFoundItem 的 found_time
found_location：继承自 RawFoundItem 的 found_location
bound_confidence：检测模型裁剪该物品区域的置信度
category：可选的单个物品分类，当前可能是多余属性
embedding_registered：该物品是否已经完成图像向量登记
registered_at：该 LostItem 写入 found_items.json 的时间

6. data/generated/image_embeddings.json

用于存储图像向量化流程生成的向量数据。

当前 embedding_engine 会把每个 item_id 对应的图像向量、图片路径、模型信息和更新时间写入该文件。

该文件是当前版本实际使用的图像向量库。

五、贡献与协作规范

1. 如果不熟悉 Git 使用，请先找组长，不要乱上传 GitHub。上手确实不容易，组长最多也只会站在桌子上生闷气。

2. 尽量保持文件目录整齐，不要添加多余文件。测试脚本和中间产物，尤其是切割后的图片，请务必加入 .gitignore，不要混入源代码上传。

3. 文件名和提交记录必须使用英文。代码注释建议使用中英双语，最后提交前删除中文注释。英语好的同学可以自行判断。

4. 由于团队内部开发环境不同，请勿直接将 pip freeze 的全部结果覆盖到 requirements.txt 中。

5. 如需引入新的第三方 Python 包，请手动将其追加至 requirements.txt，并注明合理的版本区间。
