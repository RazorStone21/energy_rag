"""Web 接口层：把问答与入库包装成事件流，并托管 front/ 下的前端页面。

这里不导出 create_app，避免导入本包时连带导入 FastAPI；
请按 src.server.app / src.server.service 直接引用需要的模块。
"""
