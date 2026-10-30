"""Explicit host checkpoint fixture for retained graph/CLI behavior tests.

Memory is a non-durable test adapter, not a supported standalone backend. PG
persistence/tenant tests live in delivery-api, the actual execution host.
"""
from langgraph.checkpoint.memory import MemorySaver
from spec_generator.graph import checkpoint_serde
from spec_generator.worker_runtime import JobRuntime,job_runtime
from spec_generator.model_config import StageModel


def hosted_runtime(root):
    from spec_generator import model_config
    models={stage:model_config.for_stage(stage) for stage in model_config._STAGE_DEFAULTS}
    runtime=JobRuntime(models,str(root/'host-calls.jsonl'),lambda text:None)
    runtime.settings['SPEC_GENERATOR_BULK']='1'
    runtime.bind_worker(owner_id='test-owner',task_id='test-task',job_id='test-job',generation=1,session_root=root)
    savers={}
    def factory(namespace):
        if namespace not in savers:
            savers[namespace]=MemorySaver(serde=checkpoint_serde());savers[namespace].setup=lambda:None
        return savers[namespace]
    runtime.checkpoint_factory=factory
    return runtime
