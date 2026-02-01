import pytest
from datetime import datetime, UTC, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, Model, Observation, Summary
from summarize import chunk_observations, get_observations_by_model, STEP


@pytest.fixture
def session():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    for name in ['self', 'user', 'system']:
        sess.add(Model(name=name, is_base=True))
    sess.commit()
    yield sess
    sess.close()


def test_get_observations_by_model(session):
    model = session.query(Model).filter_by(name='user').first()
    
    session.add(Observation(text='Obs 1', timestamp=datetime.now(UTC), model_id=model.id))
    session.add(Observation(text='Obs 2', timestamp=datetime.now(UTC), model_id=model.id))
    session.add(Observation(text='Obs 3', timestamp=datetime.now(UTC)))
    session.commit()
    
    by_model = get_observations_by_model(session)
    assert model.id in by_model
    assert len(by_model[model.id]) == 2


def test_chunk_observations_small(session):
    obs = Observation(text='Short observation', timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    
    chunks = chunk_observations([obs])
    assert len(chunks) == 1
    assert len(chunks[0]) == 1


def test_step_constant():
    assert STEP == 10
