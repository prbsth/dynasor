"""
Adopted from vLLM openai server
"""

import asyncio
import atexit
import gc
import importlib
import inspect
import multiprocessing
import os
import re
import signal
import socket
import sys
import tempfile
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncIterator, Dict, Optional, Set, Tuple, Union

import uvloop
import vllm.envs as envs
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingChatRequest,
                                              EmbeddingCompletionRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse,
                                            #   LoadLoraAdapterRequest,
                                              PoolingChatRequest,
                                              PoolingCompletionRequest,
                                              PoolingRequest, PoolingResponse,
                                              RerankRequest, RerankResponse,
                                              ScoreRequest, ScoreResponse,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                            #   UnloadLoraAdapterRequest
                                            )
from vllm.entrypoints.openai.protocol import CompletionStreamResponse
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
# yapf: enable

from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.serving_rerank import JinaAIServingRerank
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (
    FlexibleArgumentParser,
    get_open_zmq_ipc_path,
    is_valid_ipv6_address,
    set_ulimit,
)
from vllm.version import __version__ as VLLM_VERSION

from dynasor.cli.serving_chat import OpenAIServingChat
from dynasor.cli.serving_completion import AdaptiveComputeOpenAIServingCompletion
from dynasor.core.entropy import (
    obtain_answer,
    should_early_exit,
    uncertain_words,
    is_certain_answer,
)

TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger("vllm.entrypoints.openai.api_server")

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.0)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        gc.collect()
        gc.freeze()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(args: Namespace) -> AsyncIterator[EngineClient]:
    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
        engine_args, args.disable_frontend_multiprocessing
    ) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # AsyncLLMEngine.
    if (
        MQLLMEngineClient.is_unsupported_config(engine_args)
        or envs.VLLM_USE_V1
        or disable_frontend_multiprocessing
    ):

        engine_client: Optional[EngineClient] = None
        try:
            engine_client = AsyncLLMEngine.from_engine_args(
                engine_args=engine_args, usage_context=UsageContext.OPENAI_API_SERVER
            )
            yield engine_client
        finally:
            if engine_client and hasattr(engine_client, "shutdown"):
                engine_client.shutdown()

    # MQLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup."
            )

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.debug("Multiprocessing frontend to use %s for IPC Path.", ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value("b", True, lock=False)
        engine_process = context.Process(
            target=run_mp_engine,
            args=(engine_args, UsageContext.OPENAI_API_SERVER, ipc_path, engine_alive),
        )
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        def _cleanup_ipc_path():
            socket_path = ipc_path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

        # Ensure we clean up the local IPC socket file on exit.
        atexit.register(_cleanup_ipc_path)

        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_config = engine_args.create_engine_config()
        build_client = partial(MQLLMEngineClient, ipc_path, engine_config, engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client
        )
        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    if not engine_process.is_alive() or not engine_alive.value:
                        raise RuntimeError(
                            "Engine process failed to start. See stack "
                            "trace for the root cause."
                        ) from None

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mq_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess

            multiprocess.mark_process_dead(engine_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.debug(
            "vLLM to use %s as PROMETHEUS_MULTIPROC_DIR", prometheus_multiproc_dir_path
        )
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


def adaptive_compute_completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.adaptive_compute_openai_serving_completion


def pooling(request: Request) -> Optional[OpenAIServingPooling]:
    return request.app.state.openai_serving_pooling


def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding


def score(request: Request) -> Optional[OpenAIServingScores]:
    return request.app.state.openai_serving_scores


def rerank(request: Request) -> Optional[JinaAIServingRerank]:
    return request.app.state.jinaai_serving_reranking


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.api_route("/ping", methods=["GET", "POST"])
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)


@router.post("/tokenize")
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_tokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    generator = await handler.create_detokenize(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


async def handle_adaptive_compute(request: CompletionRequest, raw_request: Request):
    handler = adaptive_compute_completion(raw_request)
    assert handler is not None

    original_request = request.model_copy()
    adaptive_compute = request.adaptive_compute

    # Setup basic configurations (will be updated)
    remaining_tokens = request.max_tokens
    sending_prompt = request.prompt
    answers = []
    is_certains = []
    output_var_template: Optional[CompletionStreamResponse] = None

    # Setup basic constants
    probe_text = adaptive_compute.get("probe_text", None)
    probe_text_end = adaptive_compute.get("probe_text_end", None)
    certainty_window = adaptive_compute.get("certainty_window", 2)
    token_interval = adaptive_compute.get("token_interval", 32)

    assert probe_text is not None

    # Setup the request for actual query.
    while remaining_tokens > 0:
        print(f"{repr(sending_prompt) = }")

        request.prompt = sending_prompt
        request.stream = True
        request.max_tokens = min(token_interval, remaining_tokens)
        request.temperature = original_request.temperature
        request.top_p = original_request.top_p
        request.adaptive_compute = None

        generator = await handler.create_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            error_response = generator.model_dump()
            logger.error(f"Error response from handler: {error_response}")
            yield f"data: {error_response}\n\n"
            yield f"data: [DONE]\n\n"
            return

        # Send the actual query
        output_text = ""
        async for output in generator:
            token = output.choices[0].text
            output_text += token
            remaining_tokens -= 1
            response_json = output.model_dump_json(
                exclude_unset=False, exclude_none=True
            )
            if output_var_template is None:
                output_var_template = output
            yield f"data: {response_json}\n\n"

        # Send the probe query
        sending_prompt += output_text
        probe_sending_prompt = sending_prompt + probe_text
        request.prompt = probe_sending_prompt
        request.stream = True
        request.max_tokens = 20
        request.temperature = 0.6
        request.top_p = 0.95
        request.adaptive_compute = None

        # TODO: Refactor this to use non-async streaming API.
        probe_generator = await handler.create_completion(request, raw_request)
        output_text = ""
        async for output in probe_generator:
            # print(f"probe_output = {output}")
            # print(f"{type(output) = }")
            token = output.choices[0].text
            output_text += token

        # Get the answer from the probe response
        answer = obtain_answer(output_text)
        answers.append(answer)

        # Check if the answer is certain
        is_certain = is_certain_answer(output_text, uncertain_words)
        is_certains.append(is_certain)

        if should_early_exit(
            answers, output_text, uncertain_words, certainty_window, is_certains
        ):
            logger.debug(f"Early exit on answer: {answer = }")

            response_obj = output_var_template.model_copy()
            response_obj.choices[0].text = probe_text + answer + probe_text_end
            response_obj.choices[0].finish_reason = "stop"
            response_obj.choices[0].stop_reason = "stop"

            response_json = response_obj.model_dump_json(
                exclude_unset=False, exclude_none=True
            )
            yield f"data: {response_json}\n\n"
            yield f"data: [DONE]\n\n"
            break

        if remaining_tokens <= 0:
            logger.debug(f"Early exit: Remaining tokens <= 0")
            yield f"data: [DONE]\n\n"
            break

    yield f"data: [DONE]\n\n"


@router.post("/v1/completions")
@with_cancellation
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)

    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )

    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)

    if adaptive_compute is not None:
        generator = handle_adaptive_compute(request, raw_request)
        return StreamingResponse(content=generator, media_type="text/event-stream")

    print("Handling normal completion")
    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/chat/completions")
@with_cancellation
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )

    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)

    if adaptive_compute is not None:
        newprompt=await handler.get_completion_prompt(request, raw_request)
        request = CompletionRequest(prompt=newprompt,**json_obj)
        generator = handle_adaptive_compute(request, raw_request)
        return StreamingResponse(content=generator, media_type="text/event-stream")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
@with_cancellation
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        fallback_handler = pooling(raw_request)
        if fallback_handler is None:
            return base(raw_request).create_error_response(
                message="The model does not support Embeddings API"
            )

        logger.warning(
            "Embeddings API will become exclusive to embedding models "
            "in a future release. To return the hidden states directly, "
            "use the Pooling API (`/pooling`) instead."
        )

        res = await fallback_handler.create_pooling(request, raw_request)

        generator: Union[ErrorResponse, EmbeddingResponse]
        if isinstance(res, PoolingResponse):
            generator = EmbeddingResponse(
                id=res.id,
                object=res.object,
                created=res.created,
                model=res.model,
                data=[
                    EmbeddingResponseData(
                        index=d.index,
                        embedding=d.data,  # type: ignore
                    )
                    for d in res.data
                ],
                usage=res.usage,
            )
        else:
            generator = res
    else:
        generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/pooling")
@with_cancellation
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Pooling API"
        )

    generator = await handler.create_pooling(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, PoolingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/score")
@with_cancellation
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Score API"
        )

    generator = await handler.create_score(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/score")
@with_cancellation
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly."
    )

    return await create_score(request, raw_request)


@router.post("/rerank")
@with_cancellation
async def do_rerank(request: RerankRequest, raw_request: Request):
    handler = rerank(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Rerank (Score) API"
        )
    generator = await handler.do_rerank(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, RerankResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/rerank")
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request):
    logger.warning_once(
        "To indicate that the rerank API is not part of the standard OpenAI"
        " API, we have located it at `/rerank`. Please update your client"
        "accordingly. (Note: Conforms to JinaAI rerank API)"
    )

    return await do_rerank(request, raw_request)


@router.post("/v2/rerank")
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request):
    return await do_rerank(request, raw_request)


TASK_HANDLERS: Dict[str, Dict[str, tuple]] = {
    "generate": {
        "messages": (ChatCompletionRequest, create_chat_completion),
        "default": (CompletionRequest, create_completion),
    },
    "embed": {
        "messages": (EmbeddingChatRequest, create_embedding),
        "default": (EmbeddingCompletionRequest, create_embedding),
    },
    "score": {"default": (RerankRequest, do_rerank)},
    "rerank": {"default": (RerankRequest, do_rerank)},
    "reward": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
    "classify": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
}

if envs.VLLM_SERVER_DEV_MODE:

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request: Request):
        """
        Reset the prefix cache. Note that we currently do not check if the
        prefix cache is successfully reset in the API server.
        """
        logger.info("Resetting prefix cache...")
        await engine_client(raw_request).reset_prefix_cache()
        return Response(status_code=200)


@router.post("/invocations")
async def invocations(raw_request: Request):
    """
    For SageMaker, routes requests to other handlers based on model `task`.
    """
    body = await raw_request.json()
    task = raw_request.app.state.task

    if task not in TASK_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: '{task}' for '/invocations'. "
            f"Expected one of {set(TASK_HANDLERS.keys())}",
        )

    handler_config = TASK_HANDLERS[task]
    if "messages" in body:
        request_model, handler = handler_config["messages"]
    else:
        request_model, handler = handler_config["default"]

    # this is required since we lose the FastAPI automatic casting
    request = request_model.model_validate(body)
    return await handler(request, raw_request)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!"
    )

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


# if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
#     logger.warning(
#         "Lora dynamic loading & unloading is enabled in the API server. "
#         "This should ONLY be used for local development!"
#     )

#     @router.post("/v1/load_lora_adapter")
#     async def load_lora_adapter(request: LoadLoraAdapterRequest, raw_request: Request):
#         handler = models(raw_request)
#         response = await handler.load_lora_adapter(request)
#         if isinstance(response, ErrorResponse):
#             return JSONResponse(
#                 content=response.model_dump(), status_code=response.code
#             )

#         return Response(status_code=200, content=response)

#     @router.post("/v1/unload_lora_adapter")
#     async def unload_lora_adapter(
#         request: UnloadLoraAdapterRequest, raw_request: Request
#     ):
#         handler = models(raw_request)
#         response = await handler.unload_lora_adapter(request)
#         if isinstance(response, ErrorResponse):
#             return JSONResponse(
#                 content=response.model_dump(), status_code=response.code
#             )

#         return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(
            message=str(exc), type="BadRequestError", code=HTTPStatus.BAD_REQUEST
        )
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path) :]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    if args.enable_request_id_headers:
        logger.warning(
            "CAUTION: Enabling X-Request-Id headers in the API Server. "
            "This can harm performance at high QPS."
        )

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    return app


async def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    resolved_chat_template = load_chat_template(args.chat_template)
    logger.info("Using supplied chat template:\n%s", resolved_chat_template)

    # LoRA support disabled – do not pass lora_modules and skip static-LoRA init
    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=None,
        prompt_adapters=args.prompt_adapters,
    )
    # await state.openai_serving_models.init_static_loras()
    state.openai_serving_chat = (
        OpenAIServingChat(
            engine_client,
            model_config,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            enable_reasoning=args.enable_reasoning,
            reasoning_parser=args.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        )
        if model_config.runner_type == "generate"
        else None
    )

    # OpenAIServingCompletion
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        )
        if model_config.runner_type == "generate"
        else None
    )

    state.adaptive_compute_openai_serving_completion = (
        AdaptiveComputeOpenAIServingCompletion(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        )
        if model_config.runner_type == "generate"
        else None
    )
    state.openai_serving_pooling = (
        OpenAIServingPooling(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        if model_config.runner_type == "pooling"
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        if model_config.task == "embed"
        else None
    )
    state.openai_serving_scores = (
        OpenAIServingScores(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if model_config.task == "score"
        else None
    )
    state.jinaai_serving_reranking = (
        JinaAIServingRerank(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if model_config.task == "score"
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )
    state.task = model_config.task


def create_server_socket(addr: Tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)

    return sock


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            # Workaround to work on macOS
            fd=sock.fileno() if sys.platform.startswith("darwin") else None,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task

    sock.close()


def main():
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))

if __name__ == "__main__":
    main()
