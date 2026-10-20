from .caching import CachingTask
from .channel import ChannelTask
from .offloading import OffloadingTask

REGISTRY = {t.id: t for t in (CachingTask, OffloadingTask, ChannelTask)}
