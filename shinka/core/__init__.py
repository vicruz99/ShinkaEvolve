# <<<<<<< HEAD
# # from .runner import EvolutionRunner, EvolutionConfig
# # from .sampler import PromptSampler
# # from .summarizer import MetaSummarizer
# # from .novelty_judge import NoveltyJudge
# # from .wrap_eval import run_shinka_eval

# # __all__ = [
# #     "EvolutionRunner",
# #     "PromptSampler",
# #     "MetaSummarizer",
# #     "NoveltyJudge",
# #     "EvolutionConfig",
# #     "run_shinka_eval",
# # ]


# # Lazy access so importing shinka.core doesn't eagerly import heavy modules.
# # Only import what is actually accessed.
# =======
# from .runner import EvolutionRunner, EvolutionConfig
# from .async_runner import AsyncEvolutionRunner
# from .sampler import PromptSampler
# from .summarizer import MetaSummarizer
# from .novelty_judge import NoveltyJudge
# from .async_novelty_judge import AsyncNoveltyJudge
# from .wrap_eval import run_shinka_eval
# from .prompt_evolver import (
#     SystemPromptEvolver,
#     SystemPromptSampler,
#     AsyncSystemPromptEvolver,
# )
# >>>>>>> upstream/main

# __all__ = [
#     "EvolutionRunner",
#     "EvolutionConfig",
#     "PromptSampler",
#     "MetaSummarizer",
#     "NoveltyJudge",
#     "AsyncNoveltyJudge",
#     "AsyncEvolutionRunner",
#     "EvolutionConfig",
#     "run_shinka_eval",
#     "SystemPromptEvolver",
#     "SystemPromptSampler",
#     "AsyncSystemPromptEvolver",
# ]

# def __getattr__(name):
#     if name == "run_shinka_eval":
#         from .wrap_eval import run_shinka_eval
#         return run_shinka_eval

#     if name in ("EvolutionRunner", "EvolutionConfig"):
#         from .runner import EvolutionRunner, EvolutionConfig
#         return {"EvolutionRunner": EvolutionRunner, "EvolutionConfig": EvolutionConfig}[name]

#     if name == "PromptSampler":
#         from .sampler import PromptSampler
#         return PromptSampler

#     if name == "MetaSummarizer":
#         from .summarizer import MetaSummarizer
#         return MetaSummarizer

#     if name == "NoveltyJudge":
#         from .novelty_judge import NoveltyJudge
#         return NoveltyJudge

#     raise AttributeError(f"module 'shinka.core' has no attribute {name!r}")


from importlib import import_module

_EXPORTS = {
    # runner.py
    "EvolutionRunner": (".runner", "EvolutionRunner"),
    "EvolutionConfig": (".runner", "EvolutionConfig"),
    
    # async_runner.py
    "AsyncEvolutionRunner": (".async_runner", "AsyncEvolutionRunner"),
    
    # sampler.py
    "PromptSampler": (".sampler", "PromptSampler"),
    
    # summarizer.py
    "MetaSummarizer": (".summarizer", "MetaSummarizer"),
    
    # novelty_judge.py
    "NoveltyJudge": (".novelty_judge", "NoveltyJudge"),
    
    # async_novelty_judge.py
    "AsyncNoveltyJudge": (".async_novelty_judge", "AsyncNoveltyJudge"),
    
    # wrap_eval.py
    "run_shinka_eval": (".wrap_eval", "run_shinka_eval"),
    
    # prompt_evolver.py
    "SystemPromptEvolver": (".prompt_evolver", "SystemPromptEvolver"),
    "SystemPromptSampler": (".prompt_evolver", "SystemPromptSampler"),
    "AsyncSystemPromptEvolver": (".prompt_evolver", "AsyncSystemPromptEvolver"),
}

__all__ = list(_EXPORTS.keys())

def __getattr__(name: str):
    try:
        mod_path, attr = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(f"module 'shinka.core' has no attribute {name!r}") from e
    
    module = import_module(mod_path, package=__name__)
    return getattr(module, attr)

def __dir__():
    return sorted(__all__)