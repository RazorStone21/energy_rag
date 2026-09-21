"""提供生成答案和描述图片的调用方法，首次使用时才加载对应模型。"""

from __future__ import annotations

import threading


class Generator:
    def __init__(self, settings, loader=None):
        """保存文本生成配置与加载函数，准备延迟加载模型。"""
        self.settings = settings
        self._loader = loader
        self._model = None
        # 预热线程和首个请求可能几乎同时进来，加载要串行，否则同一份权重会被加载两次。
        self._lock = threading.Lock()

    def load(self):
        """第一次调用时加载文本模型，之后使用已加载的模型；测试时可替换加载函数。

        加载全程持锁：并发的第二个调用会等第一个加载完再复用同一个实例。
        """
        with self._lock:
            if self._model is None:
                loader = self._loader
                if loader is None:
                    from .local_qwen import load_llm

                    loader = load_llm
                self._model = loader(self.settings)
            return self._model

    def generate(self, prompt):
        """把提示词交给文本模型，从返回的字符串或消息对象中取出答案并去掉首尾空白。"""
        result = self.load().invoke(prompt)
        if hasattr(result, "content"):
            answer = result.content
        else:
            answer = str(result)
        return answer.strip()

    def generate_stream(self, prompt):
        """逐块产出答案文本，供调用方边生成边显示。

        模型提供 stream 时按其节奏产出；没有该方法时退回一次产出完整答案，
        保证替换模型加载函数后仍能走通流式调用。
        """
        model = self.load()
        stream = getattr(model, "stream", None)
        if stream is None:
            yield self.generate(prompt)
            return
        yield from stream(prompt)

    def release(self):
        """释放本组件持有的文本生成模型引用。"""
        with self._lock:
            self._model = None


class VisionGenerator:
    def __init__(self, settings, loader=None):
        """保存图片描述所需配置与可替换的模型加载函数。"""
        self.settings = settings
        self._loader = loader
        self._model = None
        # 与文本模型同理：入库前的预热或并发描述都可能同时触发加载。
        self._lock = threading.Lock()

    @property
    def available(self):
        """用 config.json 是否存在排除空占位目录；不在这里验证权重完整性或加载模型。"""
        return (self.settings.path / "config.json").is_file()

    def load(self):
        """首次描述图片时加载视觉模型，其后复用缓存实例。

        加载全程持锁：并发的第二个调用会等第一个加载完再复用同一个实例。
        """
        with self._lock:
            if self._model is None:
                loader = self._loader
                if loader is None:
                    from .local_qwen import load_vlm

                    loader = load_vlm
                self._model = loader(self.settings)
            return self._model

    def describe(self, image, prompt):
        """把图片和描述要求交给视觉模型，返回图片的文字说明。"""
        return self.load().describe(image, prompt)

    def describe_batch(self, images, prompts):
        """一次描述多张图片，按输入顺序返回每张的文字说明。"""
        return self.load().describe_batch(images, prompts)

    def release(self):
        """释放本组件持有的视觉模型引用。"""
        with self._lock:
            self._model = None
