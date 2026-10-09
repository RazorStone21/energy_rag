"""加载本地 Qwen 模型，完成文本生成和图片描述。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

# 流式生成时等待下一个词元的上限（秒）。预填充长提示词和最慢的一步生成都在这个范围内。
STREAM_TOKEN_TIMEOUT = 120
# 判定超时后等待生成线程收尾的时间（秒）；线程可能已经卡死，不能在这里无限等待。
STREAM_THREAD_JOIN_TIMEOUT = 5


class _QwenLLM(LLM):
    """Qwen 文本模型的 LangChain LLM 封装，具体型号由配置里的模型目录决定。

    根据 GenerationSettings.enable_thinking 控制聊天模板的思考模式。
    继承 LangChain LLM 后可通过 invoke 调用，也可接入 RAGAS 的模型包装器。
    """

    _model = PrivateAttr()
    _tokenizer = PrivateAttr()
    _settings = PrivateAttr()

    def __init__(self, model, tokenizer, settings):
        """保存已加载的语言模型、分词器和生成参数，接入 LangChain 接口。"""
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings

    @property
    def _llm_type(self) -> str:
        """向 LangChain 提供当前语言模型的类型标识。

        取模型目录名而不是写死型号：[generation] 的 path 可以改，写死的名字
        会和实际加载的模型对不上。
        """
        return self._settings.path.name

    def _prepare(self, prompt: str):
        """套用聊天模板并编码提示词，返回模型输入和本轮生成参数。

        生成参数在流式与非流式路径间共用，温度大于 0 才开启随机采样。
        """
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._settings.enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=self._settings.max_new_tokens,
            do_sample=self._settings.temperature > 0,
            repetition_penalty=1.05,
        )
        # 关闭采样时 temperature / top_p 不参与生成，传进去会被判为无效参数并告警。
        if self._settings.temperature > 0:
            gen_kwargs["temperature"] = self._settings.temperature
            gen_kwargs["top_p"] = self._settings.top_p
        return inputs, gen_kwargs

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """给提示词套用聊天模板，调用本地模型并返回新增的答案文本。

        stop、run_manager 和 kwargs 用于兼容接口；当前没有实现停止词处理。
        """
        inputs, gen_kwargs = self._prepare(prompt)

        # 这里只做推理，不记录训练所需的梯度，减少额外内存开销。
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        # 只解码模型新增的词元，避免把原始问题与上下文重复当作答案返回。
        answer = self._tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return answer.strip()

    def stream(self, prompt: str):
        """逐块产出新生成的答案文本，供调用方边生成边显示。

        generate 会阻塞到全部生成结束，因此放到后台线程执行；
        主线程从 TextIteratorStreamer 迭代取出新增词元，skip_prompt 保证不重复首部提示词。
        生成失败时异常发生在后台线程里，这里把它转成主线程可见的 RuntimeError：
        否则调用方会一直等一个再也不会产出内容的队列，Web 服务里表现为锁不释放。
        """
        import queue
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs, gen_kwargs = self._prepare(prompt)
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            # 不传超时时队列默认永久阻塞，generate 抛错后消费者再也醒不过来。
            timeout=STREAM_TOKEN_TIMEOUT,
        )
        # 后台线程不能直接把异常抛给主线程，先把失败原因放在列表里，由主线程重新抛出。
        failure: list[BaseException] = []

        def run_generate():
            """在后台线程中执行生成，失败时记录原因并结束 streamer。"""
            try:
                with torch.no_grad():
                    self._model.generate(**inputs, streamer=streamer, **gen_kwargs)
            except BaseException as exc:  # 显存不足、输入过长等都从这里退出
                failure.append(exc)
                # generate 正常结束时自己会调用 end()；异常路径必须补上，否则主线程永久等待。
                streamer.end()

        thread = Thread(target=run_generate)
        thread.start()
        timed_out = False
        try:
            yield from streamer
        except queue.Empty:
            # 超时说明后台线程既不产出也不结束，交给调用方处理，不再无限等待。
            timed_out = True
        finally:
            if timed_out:
                thread.join(timeout=STREAM_THREAD_JOIN_TIMEOUT)
            else:
                # 调用方提前停止迭代时同样等待线程收尾，避免生成任务悬挂。
                thread.join()

        if timed_out:
            raise RuntimeError(f"等待生成超过 {STREAM_TOKEN_TIMEOUT} 秒仍无新内容，已中止本轮")
        if failure:
            raise RuntimeError("生成失败，未能产出完整答案") from failure[0]


class _QwenVL:
    """Qwen-VL 封装：提供 .describe(image, prompt) -> str 描述图表/图片。

    image 为解析器提供的 PIL.Image，用于文档入库时生成图片描述。
    模型实例由 Runtime 的 VisionGenerator 持有。
    """

    def __init__(self, model, processor, settings):
        """保存已加载的视觉模型、处理器和图片描述生成参数。

        decoder-only 模型批量生成必须用左填充：右填充时每条序列的末尾是填充位，
        生成会从填充之后开始，整批可能返回空结果，而耗时会因为「其实没生成」
        显得异常快，很容易被误当成批量带来的加速。
        """
        self._model = model
        self._processor = processor
        self._settings = settings
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _prepare(self, images, prompts):
        """把若干组「图片 + 指令」编码成一次批量生成所需的输入。"""
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image, prompt in zip(images, prompts)
        ]
        texts = [
            self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        inputs = self._processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self._model.device)

    def describe(self, image, prompt: str) -> str:
        """描述单张图片，走与批量相同的代码路径。"""
        return self.describe_batch([image], [prompt])[0]

    def describe_batch(self, images, prompts) -> list[str]:
        """一次描述多张图片，按输入顺序返回每张的描述。

        多张图共用一次前向：视觉编码和提示词预填充只做一遍，实测单张耗时
        从 5.3 秒降到 1.0 秒（batch_size=12）。生成仍然是逐词元的，
        所以一批的总耗时由其中最长的那条描述决定。
        """
        if not images:
            return []
        inputs = self._prepare(images, prompts)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._settings.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        # 左填充保证所有序列的生成段都从同一位置开始，可以一起切出来。
        answers = self._processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )
        return [answer.strip() for answer in answers]


def _bnb_4bit_config():
    """加载时使用的 4bit 量化参数：NF4 压缩权重、BF16 用于计算，双重量化进一步减少存储开销。"""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _quantization_kwargs(directory):
    """按模型目录拼出加载时要传的量化参数，目录自带量化配置时返回空字典。

    预量化检查点（仓库里直接存了 4bit 权重，如 unsloth 的 bnb-4bit 仓库）的
    config.json 里已经有 quantization_config，交给它自己生效即可。这里必须整个
    省略这个参数而不能传 quantization_config=None：显式传 None 会被当成"本次不要
    量化"，transformers 按未量化构建模型，随后与检查点里的打包 4bit 权重尺寸不符，
    加载直接报错。
    """
    config_file = Path(directory) / "config.json"
    try:
        saved = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 读不到 config.json 时不在这里下结论：按项目默认的加载时量化处理，
        # 目录本身有问题会由 from_pretrained 报出，不会被静默忽略。
        saved = {}
    if saved.get("quantization_config"):
        return {}
    return {"quantization_config": _bnb_4bit_config()}


def load_llm(settings):
    """从本地目录加载文本模型和分词器，返回兼容 LangChain 的生成对象。

    settings 是 Runtime 传入的 GenerationSettings，包含模型路径和生成参数。
    本函数负责加载；最大生成长度、温度和思考模式在 _QwenLLM._call 中使用。
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(settings.path),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(settings.path),
        device_map="auto",
        trust_remote_code=True,
        **_quantization_kwargs(settings.path),
    )
    return _QwenLLM(model, tokenizer, settings)


def load_vlm(settings):
    """从本地目录加载视觉语言模型和处理器，返回图片描述对象。

    settings 是 Runtime 传入的 VisionSettings，包含模型路径和图片描述参数。
    模型负责结合图片与文字生成描述，处理器负责准备输入及解码输出。
    """
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    # 视觉处理器同时准备图片和文本输入，需与模型权重配套。
    processor = AutoProcessor.from_pretrained(
        str(settings.path),
        trust_remote_code=True,
    )
    # 按目录里 config.json 的 model_type 选具体类（Qwen3-VL、Qwen2.5-VL 都能加载），
    # 换视觉模型时不必再改这里的类名。
    model = AutoModelForImageTextToText.from_pretrained(
        str(settings.path),
        device_map="auto",
        trust_remote_code=True,
        **_quantization_kwargs(settings.path),
    )
    return _QwenVL(model, processor, settings)
