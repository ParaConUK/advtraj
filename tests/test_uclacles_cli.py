import tempfile
from pathlib import Path

import advtraj.cli.uclales


def test_uclales_cli(uclales_testdata_path):
    # TODO: move prefix somewhere else
    file_prefix = "rico"
    with tempfile.TemporaryDirectory() as tempdir:
        advtraj.cli.uclales.main(
            data_path=uclales_testdata_path,
            file_prefix=file_prefix,
            output_path=str(Path(tempdir) / "trajectories.nc"),
        )
