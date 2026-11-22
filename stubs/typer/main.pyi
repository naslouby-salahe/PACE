from collections.abc import Sequence

from typer import Typer

class Command:
    def main(
        self,
        args: Sequence[str] | None = ...,
        prog_name: str | None = ...,
        standalone_mode: bool = ...,
    ) -> int | None: ...

def get_command(typer_instance: Typer) -> Command: ...
