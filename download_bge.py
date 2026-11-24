r"""仅用于一次性从 ModelScope 获取 BGE 离线包；后续推理无需此脚本。"""
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

ROOT = Path(r"D:\xiangmu-dianying")
if __name__ == "__main__":
    model_dir = snapshot_download(
        model_id="AI-ModelScope/bge-base-en-v1.5",
        local_dir=str(ROOT / "bge-base-en-v1.5"),
        cache_dir=str(ROOT / "work" / "modelscope_cache"),
        revision="master",
        # CPU PyTorch 离线推理只需一份权重；排除重复框架导出，减少下载体积。
        ignore_patterns=["*.onnx", "onnx/*", "*.msgpack", "*.h5", "*.ot"],
    )
    print("Model saved:", model_dir)
