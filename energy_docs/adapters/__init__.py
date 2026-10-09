"""站点适配器。

通用适配器 `SiteAdapter` 处理绝大多数中文政府 CMS；只有遇到不兼容的站点
才需要在这里新增专用适配器。
"""

from .base import CrawlResult, SiteAdapter, SiteSpec, build_page_urls

__all__ = ["CrawlResult", "SiteAdapter", "SiteSpec", "build_page_urls"]
