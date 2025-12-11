import bpy
import inspect

from bpy.types import Context, Event
from typing import Set

from ..jobs import BackgroundJob
from ...logger import LOGGER


def process_operator(cls):
    """
    Decorator for operators that need to do a lot of computation.
    Requires setup() to do all initial computations that require bpy context,
    process() to do all data processing with no access to Blender data on a different thread,
    coalesce() to use the result from process() and bring it back to Blender
    rescind() to clean up in case of failure or cancellation.
    """

    required = ["setup", "process", "coalesce", "rescind"]

    for r in required:
        if not hasattr(cls, r) or not callable(getattr(cls, r)):
            raise TypeError(f"{cls.__name__} missing required method: {r}()")

        if r != "process":
            # Check that the method has 'context' as first argument
            sig = inspect.signature(getattr(cls, r))
            params = list(sig.parameters.values())
            if (
                len(params) < 2
                or params[0].name != "self"
                or params[1].name != "context"
            ):
                raise TypeError(
                    f"{cls.__name__}.{r}() must have 'self' and 'context' as its first arguments."
                )

    def execute(self, context: Context) -> Set[str]:
        try:
            LOGGER.debug("Setup initialized.")
            self.setup(context)
        except Exception as e:
            LOGGER.error(f"Setup failed: {e}", exc=e)
            return self.clean(context)

        self._job = BackgroundJob(self.process)
        LOGGER.info("Processing job initialized.")
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: Context, event: Event) -> Set[str]:
        if event.type == "ESC":
            self.clean(context)
        if event.type == "TIMER" and self._job.is_done():
            self._result, exc = self._job.get_result()
            if isinstance(exc, Exception):
                LOGGER.error(f"Process failed: {exc}", exc=exc)
                return self.clean(context)
            try:
                self.coalesce(context)
            except Exception as e:
                LOGGER.error(f"Failed to coalesce results: {e}", exc=e)
                return self.clean(context)
            LOGGER.debug("Process complete.")
            LOGGER.coalesce(self)
            return {"FINISHED"}
        LOGGER.coalesce(self)
        return {"PASS_THROUGH"}

    def clean(self, context: Context) -> Set[str]:
        self._job = None
        try:
            context.window_manager.event_timer_remove(self._timer)
            self.rescind(context)
        finally:
            LOGGER.coalesce(self)
            return {"CANCELLED"}

    cls.execute = execute
    cls.modal = modal
    cls.clean = clean

    return cls
