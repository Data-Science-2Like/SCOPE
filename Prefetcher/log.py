from pathlib import Path
from typing import Optional

LOGFILE = None


def log(*print_args, logfile : Optional[Path] = None) -> None:
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        if not logfile.parent.exists():
            logfile.parent.mkdir(exist_ok=True, parents=True)
        with open(logfile, 'a') as f:
            print(*print_args, file=f)
    elif LOGFILE:
        if not LOGFILE.parent.exists():
            LOGFILE.parent.mkdir(exist_ok=True, parents=True)
        with open(LOGFILE, 'a') as f:
            print(*print_args, file=f)
    print(*print_args)

