"""serve subcommand."""
import click
import infscale.client as client
from infscale.constants import APISERVER_PORT, LOCALHOST


@click.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option(
    "--port", default=APISERVER_PORT, help="Controller's apiserver port number"
)
@click.argument("specfile")
def serve(host: str, port: int, specfile: str) -> None:
    """Serve model based on config yaml file."""
    client.serve(host, port, specfile)
