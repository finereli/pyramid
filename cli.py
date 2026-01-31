import click
from datetime import datetime, UTC
from db import init_db, get_session, Observation, Model


@click.group()
def cli():
    init_db()


@cli.command()
@click.argument('text')
@click.option('--importance', '-i', default=5, help='Importance 1-10+')
def observe(text, importance):
    session = get_session()
    obs = Observation(text=text, importance=importance, timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    click.echo(f'Added observation #{obs.id} (importance={importance})')
    session.close()


@cli.command('list')
@click.option('--limit', '-n', default=20, help='Number of observations to show')
def list_observations(limit):
    session = get_session()
    observations = session.query(Observation).order_by(Observation.timestamp.desc()).limit(limit).all()
    
    if not observations:
        click.echo('No observations yet.')
        return
    
    for obs in reversed(observations):
        model_name = obs.model.name if obs.model else '-'
        click.echo(f'[{obs.id}] {obs.timestamp:%Y-%m-%d %H:%M} ({obs.importance}) [{model_name}] {obs.text}')
    
    session.close()


@cli.command()
def models():
    session = get_session()
    all_models = session.query(Model).all()
    
    for m in all_models:
        base_marker = '*' if m.is_base else ' '
        desc = m.description or '(no description)'
        click.echo(f'{base_marker} {m.name}: {desc}')
    
    session.close()


if __name__ == '__main__':
    cli()
