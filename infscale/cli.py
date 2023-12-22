"""command line tool."""
import asyncio

import click

from infscale.constants import CONTROLLER_PORT
from infscale.controller import controller as ctrl
from infscale.version import VERSION


@click.group()
@click.version_option(version=VERSION)
def cli():  # noqa: D103
    pass


@cli.command()
@click.option("--port", default=CONTROLLER_PORT, help="port number")
def controller(port):
    """Run controller."""
    asyncio.run(ctrl.Controller(port=port).run())


@cli.command()
def agent():
    """Run agent."""
    print("Run agent")


if __name__ == "__main__":
    cli()
