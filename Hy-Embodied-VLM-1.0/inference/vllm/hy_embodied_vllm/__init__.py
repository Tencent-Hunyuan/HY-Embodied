"""
hy_embodied_vllm — vLLM plugin for Hy-Embodied-VLM-1.0 (Tencent Hunyuan)

Provides:
  - Model class HYV3VLForConditionalGeneration (VL, 30B-A3B MoE) via ModelRegistry
  - Model class HYV3ForCausalLM (LLM base) via ModelRegistry
  - Tool parser 'hy_v3' via ToolParserManager
  - Reasoning parser 'hunyuan_v3' via ReasoningParserManager

Auto-loaded by vLLM's plugin discovery at engine startup (see pyproject.toml
`[project.entry-points."vllm.general_plugins"]`).

Usage:
  pip install -e ./inference/vllm/       # install plugin
  vllm serve tencent/Hy-Embodied-VLM-1.0 \
      --served-model-name hy_a3b \
      --trust-remote-code \
      --reasoning-parser hunyuan_v3 \
      --tool-call-parser hy_v3 \
      --chat-template <path-to-chat_template.jinja>
"""

__version__ = "0.1.0"


def register():
    """Entry-point callable invoked by vLLM at plugin load time.

    Idempotent — safe to call multiple times per process. This satisfies the
    re-entrancy requirement in vLLM's plugin guidelines
    (https://docs.vllm.ai/en/latest/design/plugin_system.html): every
    registration site checks whether the name is already registered and
    skips if so.

    Registers model classes lazily (avoids CUDA init at plugin load), then
    registers tool/reasoning parsers.
    """
    _register_models()
    _register_parsers()


def _register_models():
    """Register model classes via <module>:<class> string form.

    Using the string form avoids importing the modeling files at plugin load
    time (which would call torch.cuda), preventing the
    `RuntimeError: Cannot re-initialize CUDA in forked subprocess` issue
    when vLLM's worker processes get spawned via fork.

    Guarded by ModelRegistry.get_supported_archs() for re-entrancy.
    """
    from vllm import ModelRegistry

    existing = set(ModelRegistry.get_supported_archs())

    if "HYV3VLForConditionalGeneration" not in existing:
        ModelRegistry.register_model(
            "HYV3VLForConditionalGeneration",
            "hy_embodied_vllm.hunyuan_v3_vision:HYV3VLForConditionalGeneration",
        )
    if "HYV3ForCausalLM" not in existing:
        ModelRegistry.register_model(
            "HYV3ForCausalLM",
            "hy_embodied_vllm.hunyuan_v3:HYV3ForCausalLM",
        )


def _register_parsers():
    """Register tool + reasoning parsers.

    Parser classes are lightweight regex/state-machine implementations, safe
    to import at plugin load time. Guarded by each manager's
    list_registered() for re-entrancy.
    """
    from vllm.tool_parsers import ToolParserManager
    from vllm.reasoning import ReasoningParserManager

    # CLI-facing names (used with `--tool-call-parser <name>` / `--reasoning-parser <name>`)
    if "hy_v3" not in ToolParserManager.list_registered():
        from hy_embodied_vllm.parsers.hy_v3_tool_parser import HYV3ToolParser
        ToolParserManager.register_module(name="hy_v3", module=HYV3ToolParser)

    if "hunyuan_v3" not in ReasoningParserManager.list_registered():
        from hy_embodied_vllm.parsers.hy_v3_reasoning_parser import HYV3ReasoningParser
        ReasoningParserManager.register_module(name="hunyuan_v3", module=HYV3ReasoningParser)
