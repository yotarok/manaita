import dataclasses
import pathlib


@dataclasses.dataclass
class StagingConfig:
    staging_path: pathlib.Path
    download_cache_path: pathlib.Path | None

    num_proc: int | None = None

    max_example_per_shard: int = 2_000
    max_num_shard: int = 90_000

    @property
    def resolved_download_cache_path(self) -> pathlib.Path:
        return (
            self.download_cache_path
            if self.download_cache_path is not None
            else self.staging_path / "downloads"
        )
