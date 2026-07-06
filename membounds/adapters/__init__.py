from membounds.adapters.reference_openai import ReferenceOpenAIAdapter

ADAPTERS = {
    "reference": ReferenceOpenAIAdapter,
}


def load_adapter(name: str, **kwargs):
    if name == "mem0":
        from membounds.adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter()
    if name == "zep":
        from membounds.adapters.zep_adapter import ZepAdapter
        return ZepAdapter()
    if name in ADAPTERS:
        return ADAPTERS[name](**kwargs)
    raise SystemExit(f"Unknown adapter: {name!r}. Available: reference, mem0, zep")
