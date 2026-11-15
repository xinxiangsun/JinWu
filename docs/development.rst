开发与文档维护
==============

多包结构
------------

``packages/jinwu`` 提供共享分析，``jinwu-ep``、``jinwu-swift``、``jinwu-fermi``、``jinwu-gw``、``jinwu-gecam``
提供仪器/任务扩展，共享 namespace。Rust 扩展 ``jinwurs`` 为可选加速。
新增逻辑前先检查现有 API；公共函数记录参数、返回、单位与方法来源，
新增可复用函数在仓库 ``REUSABLE_FUNCTIONS.md`` 登记。

分层与依赖
------------

任务包负责仪器 I/O、校准与流程编排，共享层承载任务无关的数据、统计和物理语义。
依赖从任务流程指向共享能力；新增通用算法不应反向依赖任务包。
已有 HEASoft、XSPEC 等集成保留明确边界：按需加载外部运行时，缺失时给出可操作的错误。
标准事件、PHA、RMF、ARF 等产品是任务处理与共享分析之间的主要边界。

一般源模型放共享层；仅仪器专属的前向或校准模型属于任务包。
优先复用成熟科学实现，通过显式适配接入；替换算法需说明数值、性能或科学收益。
保持公共接口兼容，迁移时在相同输入与设置下比较原实现、中间量和最终结果，
将算法修正与纯接口迁移分开验收。新抽象应有当前调用者或清楚的独立职责。

构建本手册
------------

从仓库根目录使用独立环境，安装当前六个 Python 包和锁定的文档依赖：

.. code-block:: bash

   python -m venv /tmp/jinwu-docs-env
   /tmp/jinwu-docs-env/bin/python -m pip install -r docs/requirements.txt
   /tmp/jinwu-docs-env/bin/python -m pip install \
       ./packages/jinwu ./packages/jinwu-ep ./packages/jinwu-swift \
       ./packages/jinwu-fermi ./packages/jinwu-gw ./packages/jinwu-gecam
   /tmp/jinwu-docs-env/bin/python -m sphinx -b html -W --keep-going docs docs/_build/html
   /tmp/jinwu-docs-env/bin/python -m http.server --directory docs/_build/html 8000

``.readthedocs.yaml`` 配置 Ubuntu 24.04 / Python 3.12，先安装
``docs/requirements.txt``，再安装每个子包；warning 会使构建失败。
配置依据 `Read the Docs v2 配置说明 <https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_。
实际上线还需要 RTD 项目连接该仓库和分支，不能以本地构建代替远端部署确认。

文档依赖固定 Sphinx 8.2.3，并安装 jieba 以建立中文关键词索引。
升级工具链时需要实际验证中文和英文/API 搜索，严格构建通过不能证明搜索脚本可用。

API 与导航
------------

``docs/_ext/jinwu_docs.py`` 在构建前扫描各包的 AST，生成规范定义模块的 API 页，
同时输出 ``docs/api/inventory.json`` 记录模块、公共定义和排除原因。
API 内容由当前安装源码的 autodoc 获取。第三方 vendor、纯转发与包导出不重复生成；
兼容接口在 :doc:`compatibility` 解释。

API 生成文件、构建产物、临时测试与审查证据保留在 gitignore 内；
用户维护的已跟踪测试不会因为 ignore 规则自动取消跟踪。
新增功能同时更新功能地图、方法/仪器指南与可运行示例，避免只添加 API 名称。

验证与记录
------------

文档改动检查严格构建、内部链接、代码块语法、实际接口、桌面与窄屏页面。
涉及科学行为的源码改动需要代表性真实数据流程和中间/最终产物检查，
静态检查或 API 构建不能替代这项验收。
重要里程碑在 Obsidian Progress 中记录范围、环境、证据与 AI 署名。
