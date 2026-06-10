"""Command line interface for AIKernel.Monolith."""

from __future__ import annotations

import click

from .config_generator import initialize_config
from .docker_runner import inspect_container, run_container, stop_container


@click.group()
def main() -> None:
    """Launch and manage the AIKernel.Monolith Semantic OS container."""


@main.command()
def up() -> None:
    """Start the monolith container."""
    result = run_container(detach=True)
    if result.returncode == 0:
        click.echo(result.stdout.strip())
    else:
        raise click.ClickException(result.stderr.strip() or "Failed to start AIKernel.Monolith.")


@main.command()
def init() -> None:
    """Generate local external mount directories and sample configs."""
    path = initialize_config()
    click.echo(f"Sample configuration written to {path}")


@main.command()
def status() -> None:
    """Show monolith container status."""
    result = inspect_container()
    click.echo(result.stdout.strip() or "AIKernel.Monolith is not running.")


@main.command()
def stop() -> None:
    """Stop the monolith container."""
    result = stop_container()
    if result.returncode == 0:
        click.echo(result.stdout.strip())
    else:
        click.echo(result.stderr.strip() or "AIKernel.Monolith was not running.")


if __name__ == "__main__":
    main()
