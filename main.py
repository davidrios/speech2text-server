import asyncio
import logging
import os
import tempfile
import typing as t
from asyncio import CancelledError, Queue, get_running_loop
from pathlib import Path
from queue import Empty, SimpleQueue

import ffmpeg  # type:ignore
from fastapi import FastAPI

log = logging.getLogger(__name__)

app = FastAPI()


class TranscriberBase:
    def transcribe(self, audio_file: str) -> t.AsyncGenerator[str | int | None, None]:
        raise NotImplementedError


class TranscriptionProcessor:
    def __init__(
        self,
        transcriber_factory: type[TranscriberBase],
        num_instances: int = 1,
    ):
        self.queue: Queue[tuple[int, str]] = Queue(num_instances)
        self.results: dict[int, Queue[str | int | None]] = {}
        self.id = 0
        self.transcribers = [transcriber_factory() for _ in range(num_instances)]
        self.num_instances = num_instances
        self.running_tasks = []

    async def process(self, audio_file: str):
        self.id += 1
        key = self.id
        res_queue: Queue[str | int | None] = Queue()
        self.results[key] = res_queue
        await self.queue.put((key, audio_file))
        return res_queue

    async def _process_queue(self):
        while True:
            try:
                key, item = await self.queue.get()
                transcriber = self.transcribers.pop()
                res_queue = self.results[key]
                async for res in transcriber.transcribe(item):
                    await res_queue.put(res)
                self.transcribers.append(transcriber)
                self.queue.task_done()
            except CancelledError:
                break

    def start(self):
        self.running_tasks = [
            asyncio.create_task(self._process_queue())
            for _ in range(self.num_instances)
        ]

    def stop(self):
        for running_task in self.running_tasks:
            running_task.cancel()


class ConvertedAudio:
    def __init__(self, original_audio: str):
        self.original_audio = original_audio

    def convert_file(self):
        out_file = str(self.tmpdir.name / Path("out.wav"))
        ffmpeg.input(self.original_audio).output(  # type: ignore
            out_file, acodec="pcm_s16le", ac=1, ar=16000
        ).run()  # type: ignore
        return out_file

    async def __aenter__(self) -> str:
        self.tmpdir = tempfile.TemporaryDirectory()
        loop = get_running_loop()
        return await loop.run_in_executor(None, self.convert_file)

    async def __aexit__(self, exc_type: t.Any, exc_val: t.Any, exc_tb: t.Any):
        self.tmpdir.cleanup()


if os.environ.get("ENGINE") == "openai":
    raise NotImplementedError
else:
    from whispercpp import Whisper, api

    class WhisperCppTranscriber(TranscriberBase):
        def __init__(self):
            self._queue: SimpleQueue[str | int | None] = SimpleQueue()
            self._transcript: list[str] = []
            params = (
                api.Params.from_enum(api.SAMPLING_GREEDY)
                .with_print_progress(False)
                .with_print_realtime(False)
                .with_suppress_blank(True)
                .build()
            )
            self._progress: list[t.Any] = []
            params.on_new_segment(self.on_new_segment, self._transcript)
            params.on_progress(self.on_progress, self._progress)
            self.w = Whisper.from_params(os.environ["MODEL_FILE"], params)

        def on_new_segment(self, ctx: api.Context, n_new: int, data: list[str]):
            segment = ctx.full_n_segments() - n_new
            while segment < ctx.full_n_segments():
                segment_text = ctx.full_get_segment_text(segment)
                self._queue.put(segment_text)
                data.append(segment_text)
                segment += 1

        def on_progress(self, ctx: api.Context, progress: int, data: t.Any):
            self._queue.put(progress)

        async def transcribe(
            self, audio_file: str
        ) -> t.AsyncGenerator[str | int | None, None]:
            async with ConvertedAudio(audio_file) as fname:
                loop = get_running_loop()
                future = loop.run_in_executor(None, self.w.transcribe_from_file, fname)
                while True:
                    try:
                        yield self._queue.get_nowait()
                        if future.done() and self._queue.empty():
                            yield None
                            break
                    except Empty:
                        await asyncio.sleep(0.1)
                await future

    transcribe_processor = TranscriptionProcessor(WhisperCppTranscriber, 1)


@app.on_event("startup")
async def startup_event():
    transcribe_processor.start()


@app.on_event("shutdown")
async def shutdown_event():
    transcribe_processor.stop()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


if os.environ.get("USE_GRADIO"):
    import gradio as gr

    use_async_iter = True
    try:
        from gradio.utils import SyncToAsyncIterator
    except ImportError:
        log.warn("** Your Gradio version doesn't support streaming responses. **")
        use_async_iter = False

    async def gradio_output_iterator(
        audio_mic: t.Any,
        audio_upload: t.Any,
    ) -> t.AsyncGenerator[str, None]:
        output = ""
        output += "** From microphone **\n"
        progress = 0

        async def process_queue(queue: Queue[str | int | None]) -> bool:
            nonlocal output
            nonlocal progress

            result = await queue.get()
            if result is None:
                return False

            if isinstance(result, int):
                progress = min(result, 100)
            else:
                output += result

            return True

        yield output
        if audio_mic is None:
            output += "** No audio **\n"
            yield output
        else:
            queue = await transcribe_processor.process(audio_mic)
            while True:
                if not await process_queue(queue):
                    break

                yield output + f"\n\n(Progress: {progress}%)"

        output += "\n"
        yield output

        output += "** From upload **\n"
        yield output
        if audio_upload is None:
            output += "** No audio **\n"
            yield output
        else:
            progress = 0
            queue = await transcribe_processor.process(audio_upload)
            while True:
                if not await process_queue(queue):
                    break

                yield output + f"\n\n(Progress: {progress}%)"

        progress = 100
        yield output + f"\n\n(Progress: {progress}%)"

    async def gradio_output(
        audio_mic: t.Any,
        audio_upload: t.Any,
    ) -> str:
        res = ""
        async for res in gradio_output_iterator(audio_mic, audio_upload):
            continue
        return res

    demo = gr.Interface(
        gradio_output_iterator if use_async_iter else gradio_output,
        inputs=[
            gr.Audio(source="microphone", label="", type="filepath"),
            gr.Audio(source="upload", label="", type="filepath"),
        ],
        outputs=gr.Text(label="Transcription"),
        allow_flagging="never",
    )

    app = gr.mount_gradio_app(app, demo.queue(), path="/gradio")
