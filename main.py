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


TranscriptionStreamItem = str | float | None
"""
The item type defines what the partial result is:

- `str`: A transcription snippet.
- `float`: The current progress in percentage.
- `None`: The process finished, no more results are coming.
"""


class TranscriberBase:
    def transcribe(
        self, audio_file: str
    ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
        """
        Implement a generator that streams the results of the transcription.
        """
        raise NotImplementedError


class TranscriptionProcessor:
    """
    This class is designed to transcribe audio files using the provided `TranscriberBase`-derived
    class. This class takes an audio file path as input and uses multiple instances of the
    provided transcriber to run the transcription process concurrently.

    The class constructor takes the following parameters:

        `transcriber_factory`: A class that derives from `TranscriberBase` and provides the
            `transcribe` method to perform the transcription. Multiple instances of this class
            are created to concurrently run the transcription process.

        `num_instances`: An optional parameter that specifies the number of transcriber
            instances to create.

    The `process` method takes an audio file path and returns an `asyncio.Queue` that the user
    can use to read the results of the transcription process as they are generated.

    The `_process_queue` method runs in a separate `asyncio` task for each transcriber instance
    and dequeues an audio file from the queue, gets an available transcriber instance, and
    starts the transcription process using the transcribe method of the transcriber
    instance. It then stores the results in the appropriate queue and puts the transcriber
    instance back in the pool.

    The `start` method starts the transcription process by creating a task for each transcriber
    instance to run the `_process_queue` method, and the stop method cancels all running tasks.
    """

    def __init__(
        self,
        transcriber_factory: type[TranscriberBase],
        num_instances: int = 1,
    ):
        self.queue: Queue[tuple[int, str]] = Queue(num_instances)
        self.results: dict[int, Queue[TranscriptionStreamItem]] = {}
        self.last_job_id = 0
        self.transcribers = [transcriber_factory() for _ in range(num_instances)]
        self.num_instances = num_instances
        self.running_tasks = []

    async def process(self, audio_file: str):
        self.last_job_id += 1
        key = self.last_job_id
        res_queue: Queue[TranscriptionStreamItem] = Queue()
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
            self._queue: SimpleQueue[TranscriptionStreamItem] = SimpleQueue()
            self._transcript: list[str] = []
            num_threads = int(os.environ.get("WHISPER_CPP_THREADS") or -1)
            if num_threads == -1:
                num_threads = os.cpu_count() or 4

            params = (
                api.Params.from_enum(api.SAMPLING_GREEDY)
                .with_print_progress(False)
                .with_print_realtime(False)
                .with_suppress_blank(True)
                .with_num_threads(num_threads)
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
            self._queue.put(min(progress, 100) / 100)

        async def transcribe(
            self, audio_file: str
        ) -> t.AsyncGenerator[TranscriptionStreamItem, None]:
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

    transcribe_processor = TranscriptionProcessor(
        WhisperCppTranscriber, int(os.environ.get("WHISPER_CPP_INSTANCES") or 1)
    )


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

        async def process_queue(queue: Queue[TranscriptionStreamItem]) -> bool:
            nonlocal output
            nonlocal progress

            result = await queue.get()
            if result is None:
                return False

            if isinstance(result, float):
                progress = int(result * 100)
            else:
                output += result

            return True

        yield output
        if audio_mic is None:
            output += "** No audio **"
            yield output
        else:
            queue = await transcribe_processor.process(audio_mic)
            while True:
                if not await process_queue(queue):
                    break

                yield output + f"\n\n(Progress: {progress}%)"

        output += "\n\n"
        yield output

        output += "** From upload **\n"
        yield output
        if audio_upload is None:
            output += "** No audio **"
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

    app = gr.mount_gradio_app(app, demo.queue(concurrency_count=10), path="/gradio")
